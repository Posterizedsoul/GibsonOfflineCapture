import PySpin
import cv2
import os
import ctypes
import time
import json
import uuid
import queue
import threading
import urllib.error
import urllib.request
import numpy as np
import dearpygui.dearpygui as dpg

# ==========================================
# CONFIGURATION (most of it editable in the GUI at runtime)
# ==========================================
SAVE_ROOT = r"C:\Gibson"
JETSON_URL = "https://drstreet.taildab2f8.ts.net"  # Funnel: public, no client install
API_KEY = os.environ.get("GIBSON_API_KEY", "")     # avoids retyping it every launch
ZOOM = 1.13              # calibration knob: lens/sensor crop, tune per rig
GAMMA = 1.0              # calibration knob: display only, 1.0 = raw passthrough
PREVIEW_W = 900          # texture width, keep small - these are old machines
PANEL_W = 300            # at Normal text size; scales with the preset
ROWS = 6                 # class-probability bars
FONT = cv2.FONT_HERSHEY_SIMPLEX
LIGHTS = {1: "Ring_and_Small_Lights", 2: "Only_Ring_Light", 3: "Only_Small_Lights"}

# Every size is rasterised as its own font. set_global_font_scale() stretches one
# bitmap instead, which is exactly what made the text look smeared before.
FAMILIES = {"Segoe UI": r"C:\Windows\Fonts\segoeui.ttf",
            "Tahoma": r"C:\Windows\Fonts\tahoma.ttf",
            "Consolas": r"C:\Windows\Fonts\consola.ttf"}
ROLE_PT = {"small": 13, "body": 16, "head": 20, "big": 28}
TEXT_SIZES = {"Small": 0.85, "Normal": 1.0, "Large": 1.2, "Huge": 1.45}

GREEN, AMBER, RED = (90, 215, 125), (235, 185, 70), (235, 105, 105)
HEAD_C, DIM_C = (235, 235, 240), (145, 145, 152)

# Settings live in the user's own profile, so a git pull never clobbers them and
# the API key is not in the repo. Plaintext: same trust level as the capture
# folder itself - fine for a station PC, not for a shared login.
CONF = os.path.join(os.path.expanduser("~"), ".gibson_capture.json")
REMEMBER = ("url", "key", "vendor", "grade", "start_id", "total", "task",
            "model_version", "tta", "interval", "timeout", "legacy_color", "boxes",
            "font_family", "text_size", "gamma")

S = {
    "run": True, "frame": None, "preview": None,
    "id": 0, "img": 1, "last_id": -1, "session": False, "saved": 0,
    "detect": False, "pred": None, "drawn": None, "log": [], "dark": True,
    "lut": None, "pw": PREVIEW_W, "ph": 0, "panel": PANEL_W, "src_w": 1920,
    "aspect": 0.75, "tex": None, "tex_tag": "texA", "idle": None, "idle_sub": "",
    "fonts": {}, "roles": {}, "bars": {}, "btn": {},
}
SAVE_Q = queue.Queue()   # capture never blocks the live feed


# ------------------------------------------------------------------ Jetson API
def _multipart(fields=(), files=()):
    """multipart/form-data body. fields: (name, value); files: (name, filename, bytes).
    Repeated names are how the API takes several images, so this takes lists."""
    b = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields:
        out += (f"--{b}\r\nContent-Disposition: form-data; "
                f"name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    for k, fn, data in files:
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                f"filename=\"{fn}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


# The gateway is reached directly (Funnel ingress or LAN). A machine with a
# corporate proxy configured would otherwise send that traffic to the proxy.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _base(u):
    """Tolerate what people actually paste: no scheme, trailing slash, /ui, /v1."""
    u = u.strip().rstrip("/")
    if not u.startswith(("http://", "https://")):
        # tailscale serve/funnel terminates TLS on 443, so a bare name is https
        u = ("https://" if u.endswith(".ts.net") else "http://") + u
    for tail in ("/ui", "/v1"):
        if u.endswith(tail):
            u = u[:-len(tail)]
    return u


def _tailnet(url):
    """True only for a literal 100.64.0.0/10 address, which is reachable solely
    from a machine running Tailscale.

    A .ts.net name is NOT in this set: with Funnel enabled it resolves through
    Tailscale's public ingress and works from anywhere with nothing installed.
    (On a machine that does run Tailscale, MagicDNS answers the same name with
    the CGNAT address instead - so never infer the path from a local lookup.)"""
    return any(f"//100.{n}." in url for n in range(64, 128))


def _hint(err, url=""):
    """Turn a raw exception into the thing to go and check."""
    e = err.lower()
    if "timed out" in e or "timeout" in e:
        return ("NO ROUTE to a 100.x tailnet address - that one needs Tailscale running "
                "on THIS pc. The https://<host>.ts.net address works anywhere while "
                "Funnel is on, with nothing installed." if _tailnet(url) else
                "NO ROUTE - nothing answered. Check the address; if it is a .ts.net "
                "name, confirm Funnel is still on (tailscale funnel status).")
    if "refused" in e:
        return "REFUSED - address reached but nothing listening. Right port? Gateway running?"
    if "unknown url type" in e or "no host" in e:
        return "BAD URL - needs to look like https://host.ts.net or http://192.168.x.x:8000"
    if "name or service" in e or "getaddrinfo" in e:
        return "CANNOT RESOLVE - check the hostname spelling."
    if "401" in e:
        return "KEY REJECTED - paste the whole key, it is case-sensitive."
    if "403" in e:
        return "WRONG SCOPE - this key is not allowed on that endpoint."
    return err


def _call(url, key, timeout, body=None, ctype=None):
    """Every endpoint wants X-API-Key; 401/403/422 come back as JSON detail."""
    headers = {"X-API-Key": key} if key else {}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=body, headers=headers)
    # Two attempts: over a relayed path the first TLS handshake regularly times
    # out while the second, on a warm path, is fine. Retrying only on timeout
    # keeps a real 401 or refusal instant.
    for attempt in (1, 2):
        try:
            with OPENER.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode(errors="replace")[:300]
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            return {"error": f"HTTP {ex.code}: {detail}"}
        except Exception as ex:
            err = {"error": f"{type(ex).__name__}: {ex}"}
            if attempt == 2 or "timed out" not in str(ex).lower():
                return err
    return err


def predict(base, key, bgr, task="classification", model_version="", tta=False, timeout=8.0):
    """POST /v1/predict - stateless: the server runs the active model and answers,
    nothing is stored. Field name is `images` and it may repeat for multi-view."""
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return {"error": "jpeg encode failed"}
    fields = [("task", task), ("tta", "true" if tta else "false")]
    if model_version:
        fields.append(("model_version", model_version))
    body, ctype = _multipart(fields, [("images", "frame.jpg", jpg.tobytes())])
    return _call(_base(base) + "/v1/predict", key, timeout, body, ctype)


def health(base, timeout=10.0):
    """GET /health - no key required, so it separates 'network down' from 'bad key'."""
    return _call(_base(base) + "/health", "", timeout)


def _ranked(resp):
    """[(label, prob)] best first. The server returns probs{class: p}; the other
    shapes are here so a plain detector or a bare list still renders."""
    if isinstance(resp, dict) and isinstance(resp.get("probs"), dict):
        return sorted(resp["probs"].items(), key=lambda kv: kv[1], reverse=True)
    items = resp if isinstance(resp, list) else []
    if isinstance(resp, dict):
        for k in ("predictions", "detections", "results", "objects"):
            if isinstance(resp.get(k), list):
                items = resp[k]
                break
        else:
            if resp.get("label"):
                return [(str(resp["label"]), float(resp.get("confidence") or 0))]
    return sorted(((_label(d), _conf(d)) for d in items if isinstance(d, dict)),
                  key=lambda kv: kv[1], reverse=True)


def _label(d):
    return str(d.get("class") or d.get("label") or d.get("name") or "?")


def _conf(d):
    return float(d.get("confidence", d.get("score", d.get("conf", 0))) or 0)


def _box(d):
    """(x1,y1,x2,y2) in preview pixels, or None. Handles the 3 usual layouts."""
    if "bbox" in d and len(d["bbox"]) == 4:
        x, y, w, h = d["bbox"]
        return int(x), int(y), int(x + w), int(y + h)
    if all(k in d for k in ("x1", "y1", "x2", "y2")):
        return int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
    if all(k in d for k in ("x", "y", "width", "height")):
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    return None


def conf_color(c):
    """One rule for confidence everywhere: strong green, borderline amber, weak red."""
    return GREEN if c >= 0.75 else AMBER if c >= 0.5 else RED


def detect_loop():
    """Live detection: latest preview -> POST /v1/predict -> result. Own thread."""
    while S["run"]:
        f = S["preview"]
        if not S["detect"] or f is None:
            time.sleep(0.15)
            continue
        t = time.time()
        S["pred"] = predict(dpg.get_value("url"), dpg.get_value("key"), f,
                            dpg.get_value("task"), dpg.get_value("model_version"),
                            dpg.get_value("tta"), dpg.get_value("timeout"))
        time.sleep(max(0.0, dpg.get_value("interval") - (time.time() - t)))


def test_server():
    url = _base(dpg.get_value("url"))
    set_status(f"testing {url} ...", DIM_C)
    h = health(url, max(10.0, dpg.get_value("timeout")))
    if "error" in h:
        set_status(f"{_hint(h['error'], url)}\n[{url}/health]  {h['error']}", RED)
        return
    if not dpg.get_value("key"):
        set_status(f"Server up ({json.dumps(h)[:60]}) - now paste an API key", AMBER)
        return
    probe = S["preview"] if S["preview"] is not None else np.zeros((64, 64, 3), np.uint8)
    r = predict(url, dpg.get_value("key"), probe, dpg.get_value("task"),
                dpg.get_value("model_version"), False, dpg.get_value("timeout"))
    if "error" in r:
        set_status(f"Server up, predict failed\n{_hint(r['error'], url)}\n{r['error']}", RED)
        return
    S["pred"] = r
    save_conf()          # a key that just worked is worth remembering
    set_status(f"CONNECTED - {r.get('model_id')}:{r.get('model_version')} answered "
               f"{r.get('label')} in {r.get('latency_ms', 0):.0f} ms.  Settings saved.", GREEN)


def set_status(msg, color=DIM_C):
    dpg.configure_item("api_status", color=color)
    dpg.set_value("api_status", msg)
    log(msg.replace("\n", " | "))


# ------------------------------------------------------------------ capture queue
def save_loop():
    """Disk writes happen here so the live feed never stalls on a capture."""
    while S["run"]:
        try:
            path, bgr = SAVE_Q.get(timeout=0.3)
        except queue.Empty:
            continue
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, bgr)
            S["saved"] += 1
            log(f"saved {os.path.basename(path)}")
        except Exception as ex:
            log(f"SAVE FAILED: {ex}")
        finally:
            SAVE_Q.task_done()


# ------------------------------------------------------------------ camera
def grab(cam):
    """Returns a BGR frame. cv2.COLOR_BayerRG2RGB is an alias of BayerBG2BGR, so
    the result is BGR-ordered - that is why the old cv2.imshow preview looked right."""
    try:
        img = cam.GetNextImage(1000)
    except PySpin.SpinnakerException:
        return None
    if img.IsIncomplete():
        img.Release()
        return None
    bayer = np.frombuffer(img.GetData(), dtype=np.uint8).reshape(img.GetHeight(), img.GetWidth())
    bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)  # copies, safe to Release after
    img.Release()
    h, w = bgr.shape[:2]
    nw, nh = int(w / ZOOM), int(h / ZOOM)
    return bgr[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw]


# ------------------------------------------------------------------ session
def log(msg):
    S["log"] = (S["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-300:]
    dpg.set_value("log", "\n".join(S["log"][-60:]))
    dpg.set_value("last", S["log"][-1])


def load_conf():
    try:
        with open(CONF) as f:
            saved = json.load(f)
    except FileNotFoundError:
        return
    except Exception as ex:
        print("ignoring unreadable settings file:", ex)
        return
    for k in REMEMBER:
        if k in saved and dpg.does_item_exist(k):
            dpg.set_value(k, saved[k])
    set_gamma(g=dpg.get_value("gamma"))
    apply_look()


def save_conf():
    """Called on exit and after a connection test passes. Start ID is written as
    wherever the session got to, so a relaunch carries on instead of repeating."""
    try:
        conf = {k: dpg.get_value(k) for k in REMEMBER}
        if S["session"]:
            conf["start_id"] = S["id"]
            conf["total"] = max(1, S["last_id"] - S["id"] + 1)
        with open(CONF, "w") as f:
            json.dump(conf, f, indent=1)
    except Exception as ex:
        print("could not save settings:", ex)


def start_session():
    S["id"] = dpg.get_value("start_id")
    S["last_id"] = S["id"] + dpg.get_value("total") - 1
    S["img"], S["session"] = 1, True
    log(f"Session started: IDs {S['id']}-{S['last_id']}")


def set_detect(_=None, on=False):
    """Live AI and capturing are separate modes: the operator is either grading
    boards against the server or shooting a dataset, never both at once."""
    S["detect"] = on
    dpg.configure_item("cap_btn", enabled=not on,
                       label="CAPTURE      Q" if not on else "CAPTURE LOCKED")
    dpg.bind_item_theme("cap_btn", S["btn"][RED] if on else 0)
    dpg.configure_item("ai_btn", label="LIVE AI   ON" if on else "LIVE AI   OFF")
    dpg.bind_item_theme("ai_btn", S["btn"][GREEN] if on else 0)
    dpg.configure_item("raw_node", show=on)   # the server's own words, while predicting
    log("Live AI on - capture locked" if on else "Live AI off - capture ready")


def capture():
    """Queue the frame and return immediately - the feed keeps running."""
    if S["detect"]:                      # also covers the Q key, not just the button
        log("Turn Live AI off to capture")
        return
    if S["frame"] is None:
        log("No camera frame yet - nothing to capture")
        return
    if not S["session"]:
        start_session()      # first press just starts at the Start ID, no ceremony
    folder = os.path.join(SAVE_ROOT, dpg.get_value("vendor"), dpg.get_value("grade"), str(S["id"]))
    name = f"{time.strftime('%H-%M-%S')}_{S['img']}_{LIGHTS[S['img']]}.jpg"
    # 'Legacy colour' reproduces the original script's RGB2BGR swap before imwrite,
    # which stores R and B swapped. Off = the file matches what you see on screen.
    frame = S["frame"][..., ::-1] if dpg.get_value("legacy_color") else S["frame"]
    SAVE_Q.put((os.path.join(folder, name), frame))
    log(f"queued {S['id']}/{name}")

    S["img"] += 1
    if S["img"] > 3:
        S["img"] = 1
        S["id"] += 1
        if S["id"] > S["last_id"]:
            S["session"] = False
            log("All requested IDs processed.")


# ------------------------------------------------------------------ image
def set_gamma(_=None, g=None):
    S["lut"] = (np.linspace(0, 1, 256) ** (GAMMA if g is None else g) * 255).astype(np.uint8)


def draw_hud(img):
    """The operator's instruction goes ON the video, drawn by OpenCV: crisp at any
    size and independent of the GUI font."""
    scale = img.shape[1] / 900          # HUD keeps its proportions at any preview size
    light = LIGHTS[S["img"]].replace("_", " ").upper()
    if S["detect"]:
        top, bottom = "LIVE AI RUNNING", "CAPTURE IS LOCKED  -  TURN LIVE AI OFF TO SHOOT"
    elif S["session"]:
        top = f"BOARD {S['id']}      SHOT {S['img']} OF 3"
        bottom = f"SET LIGHTS TO {light},  THEN PRESS  Q"
    else:
        top = "READY"
        bottom = (f"PRESS  Q  TO SHOOT BOARD {dpg.get_value('start_id')}  -  "
                  f"SHOT 1 OF 3, LIGHTS {light}")
    band = img[:int(86 * scale)]
    band[:] = cv2.addWeighted(band, 0.25, np.zeros_like(band), 0, 0)   # legible on any board
    cv2.putText(img, top, (int(18 * scale), int(36 * scale)), FONT, 1.0 * scale,
                (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(img, bottom, (int(18 * scale), int(71 * scale)), FONT, 0.75 * scale,
                (150, 235, 170), max(1, int(2 * scale)), cv2.LINE_AA)


def draw_pred_overlay(img):
    """While Live AI runs the grade belongs on the video, big and sharp, not in
    13px panel text."""
    if not S["detect"] or not isinstance(S["pred"], dict) or "error" in S["pred"]:
        return
    ranked = _ranked(S["pred"])
    if not ranked:
        return
    label, c = ranked[0]
    scale = img.shape[1] / 900
    band = img[img.shape[0] - int(72 * scale):]
    band[:] = cv2.addWeighted(band, 0.25, np.zeros_like(band), 0, 0)
    cv2.putText(img, f"{label}   {c * 100:.0f}%",
                (int(18 * scale), img.shape[0] - int(24 * scale)), FONT, 1.3 * scale,
                conf_color(c)[::-1], max(1, int(3 * scale)), cv2.LINE_AA)


def placeholder(msg, sub):
    """So the video area never sits there as an unexplained black rectangle."""
    img = np.full((S["ph"], S["pw"], 3), 26, np.uint8)
    cv2.putText(img, msg, (30, S["ph"] // 2 - 8), FONT, 1.0, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (30, S["ph"] // 2 + 28), FONT, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ fonts
def dpi_aware():
    """Without this Windows bitmap-stretches the whole window on a scaled display
    and every pixel goes soft - the usual reason DPG text looks wonky."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)     # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()      # older Windows
        except Exception:
            pass


def build_fonts():
    """One crisp raster per (family, preset, role) - never a scaled bitmap.
    pixel_snapH aligns glyphs to whole pixels, which is what kills the fuzz."""
    have = {n: p for n, p in FAMILIES.items() if os.path.exists(p)}
    if not have:
        return ["Default"]
    with dpg.font_registry():
        for name, path in have.items():
            for preset, mult in TEXT_SIZES.items():
                for role, pt in ROLE_PT.items():
                    S["fonts"][(name, preset, role)] = dpg.add_font(
                        path, int(pt * mult), pixel_snapH=True)
    return ["Default"] + list(have)


def role(item, kind):
    """Tag an item so apply_look() can give it the right size."""
    S["roles"].setdefault(kind, []).append(item)
    return item


def apply_look(*_):
    """Rebind every tagged item to the chosen family and size preset."""
    fam, preset = dpg.get_value("font_family"), dpg.get_value("text_size")
    mult = TEXT_SIZES.get(preset, 1.0)
    S["panel"] = int(PANEL_W * mult)
    dpg.configure_item("side", width=S["panel"])
    if fam != "Default" and S["fonts"]:
        dpg.bind_font(S["fonts"][(fam, preset, "body")])
        for kind, items in S["roles"].items():
            for it in items:
                dpg.bind_item_font(it, S["fonts"][(fam, preset, kind)])
    else:
        dpg.bind_font(0)
        for items in S["roles"].values():
            for it in items:
                dpg.bind_item_font(it, 0)
    fit_image()


# ------------------------------------------------------------------ look
def make_theme(dark):
    bg, child, frame, btn, hov, act, txt, dim = (
        ((20, 20, 22), (32, 32, 35), (50, 50, 54), (66, 66, 70), (98, 98, 104),
         (130, 130, 136), (245, 245, 245), (145, 145, 150))
        if dark else
        ((240, 240, 242), (255, 255, 255), (228, 228, 231), (218, 218, 222), (198, 198, 202),
         (170, 170, 175), (18, 18, 20), (110, 110, 115)))
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            for k, v in ((dpg.mvStyleVar_FrameRounding, 4), (dpg.mvStyleVar_ChildRounding, 6),
                         (dpg.mvStyleVar_WindowRounding, 6), (dpg.mvStyleVar_GrabRounding, 4),
                         (dpg.mvStyleVar_TabRounding, 4), (dpg.mvStyleVar_FrameBorderSize, 1)):
                dpg.add_theme_style(k, v, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 9, 8, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12,
                                category=dpg.mvThemeCat_Core)
            for k, v in ((dpg.mvThemeCol_WindowBg, bg), (dpg.mvThemeCol_ChildBg, child),
                         (dpg.mvThemeCol_PopupBg, child), (dpg.mvThemeCol_FrameBg, frame),
                         (dpg.mvThemeCol_FrameBgHovered, hov), (dpg.mvThemeCol_FrameBgActive, act),
                         (dpg.mvThemeCol_Button, btn), (dpg.mvThemeCol_ButtonHovered, hov),
                         (dpg.mvThemeCol_ButtonActive, act), (dpg.mvThemeCol_Header, frame),
                         (dpg.mvThemeCol_HeaderHovered, hov), (dpg.mvThemeCol_HeaderActive, act),
                         (dpg.mvThemeCol_Tab, frame), (dpg.mvThemeCol_TabHovered, hov),
                         (dpg.mvThemeCol_TabActive, act), (dpg.mvThemeCol_Border, frame),
                         (dpg.mvThemeCol_Text, txt), (dpg.mvThemeCol_TextDisabled, dim),
                         (dpg.mvThemeCol_SliderGrab, txt), (dpg.mvThemeCol_SliderGrabActive, txt),
                         (dpg.mvThemeCol_CheckMark, txt), (dpg.mvThemeCol_ScrollbarBg, child),
                         (dpg.mvThemeCol_ScrollbarGrab, frame), (dpg.mvThemeCol_Separator, frame)):
                dpg.add_theme_color(k, v, category=dpg.mvThemeCat_Core)
    return t


def bar_theme(rgb):
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, rgb, category=dpg.mvThemeCat_Core)
    return t


def toggle_theme():
    S["dark"] = not S["dark"]
    dpg.bind_theme(S["dark_theme"] if S["dark"] else S["light_theme"])


def ensure_tex(want_w):
    """Re-cut the texture to roughly the size it is being displayed at.

    A fixed 900px texture stretched across a maximised window is just blur, and
    a big texture on a small window is wasted CPU every frame. Reallocating on a
    real size change costs nothing while nobody is dragging the window edge.
    Double-buffered tags so the image never points at a deleted texture."""
    want_w = int(max(320, min(want_w, S["src_w"], 1920)))
    if abs(want_w - S["pw"]) < 64:
        return
    h = max(1, int(want_w * S["aspect"]))
    S["pw"], S["ph"] = want_w, h
    S["tex"] = np.zeros(want_w * h * 3, dtype=np.float32)
    old, new = S["tex_tag"], ("texB" if S["tex_tag"] == "texA" else "texA")
    dpg.add_raw_texture(want_w, h, S["tex"], format=dpg.mvFormat_Float_rgb,
                        tag=new, parent="texreg")
    dpg.configure_item("img", texture_tag=new)
    dpg.delete_item(old)
    S["tex_tag"] = new
    S["idle"] = placeholder("NO CAMERA DETECTED", S["idle_sub"])


def fit_image(*_):
    """Scale the preview to whatever room the window currently has."""
    w = dpg.get_viewport_client_width() - S["panel"] - 70
    h = dpg.get_viewport_client_height() - 130       # tab bar + last-event line
    show_w = int(min(w, h / S["aspect"]))            # fit, preserving aspect
    ensure_tex(show_w)
    dpg.configure_item("img", width=max(160, show_w),
                       height=max(120, int(show_w * S["aspect"])))


# ------------------------------------------------------------------ ui
def section(text):
    """A titled block: heading in the head size, then a rule."""
    role(dpg.add_text(text.upper(), color=HEAD_C), "head")
    dpg.add_separator()


def note(text):
    """Small dim explanatory line."""
    return role(dpg.add_text(text, color=DIM_C, wrap=560), "small")


def btn_theme(rgb):
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgb, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgb, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgb, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (20, 20, 22), category=dpg.mvThemeCat_Core)
    return t


def build_ui(families):
    with dpg.texture_registry(tag="texreg"):
        dpg.add_raw_texture(S["pw"], S["ph"], S["tex"], format=dpg.mvFormat_Float_rgb,
                            tag=S["tex_tag"])

    S["bars"] = {c: bar_theme(c) for c in (GREEN, AMBER, RED, (95, 95, 100))}
    S["btn"] = {c: btn_theme(c) for c in (GREEN, RED)}

    with dpg.window(tag="root"):
        with dpg.tab_bar():

            # ============================================== the working screen
            with dpg.tab(label="   Capture   "):
                with dpg.group(horizontal=True):
                    dpg.add_image(S["tex_tag"], tag="img")

                    with dpg.child_window(width=S["panel"], autosize_y=True, tag="side"):
                        dpg.add_button(label="CAPTURE      Q", callback=capture,
                                       width=-1, height=66, tag="cap_btn")
                        dpg.add_button(label="LIVE AI   OFF", width=-1, height=38,
                                       tag="ai_btn",
                                       callback=lambda: set_detect(on=not S["detect"]))
                        with dpg.group(horizontal=True):
                            role(dpg.add_text("0", tag="c_saved", color=GREEN), "head")
                            note("saved")
                            role(dpg.add_text("0", tag="c_queue", color=DIM_C), "head")
                            note("queued")
                        dpg.add_spacer(height=6)

                        section("Do this now")
                        role(dpg.add_text("", tag="now_light", wrap=260, color=GREEN), "head")
                        role(dpg.add_text("", tag="now_step", wrap=260, color=DIM_C), "small")
                        dpg.add_spacer(height=6)

                        section("Picture")
                        dpg.add_slider_float(label="Gamma", tag="gamma",
                                             default_value=GAMMA, min_value=0.4,
                                             max_value=3.0, width=-70,
                                             callback=lambda s, v: set_gamma(g=v))
                        note("display only, saved files are untouched")
                        dpg.add_spacer(height=6)

                        section("Live AI")
                        dpg.add_checkbox(label="Predict continuously", tag="detect_cb",
                                         callback=set_detect)
                        dpg.add_slider_float(label="Every s", tag="interval",
                                             default_value=1.0, min_value=0.2,
                                             max_value=10.0, width=-70)
                        role(dpg.add_text("-", tag="pred_label", wrap=260, color=DIM_C), "big")
                        for i in range(ROWS):
                            dpg.add_progress_bar(tag=f"bar{i}", width=-1, overlay="", show=False)
                        note("green above 75%, amber above 50%, red below")
                        with dpg.tree_node(label="Raw response", tag="raw_node",
                                           show=False, default_open=True):
                            role(dpg.add_text("", tag="pred_raw_live", wrap=250,
                                              color=DIM_C), "small")
                role(dpg.add_text("", tag="last", color=DIM_C), "small")

            # ============================================== everything else, two columns
            with dpg.tab(label="   Settings   "):
                with dpg.group(horizontal=True):

                    # ---------------------------------- left: this station
                    with dpg.child_window(width=520, autosize_y=True):
                        section("Where the images go")
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label="Vendor", tag="vendor",
                                               default_value="Electric_Hardwood", width=200)
                            dpg.add_spacer(width=16)
                            dpg.add_input_text(label="Grade", tag="grade",
                                               default_value="2A", width=110)
                        note(SAVE_ROOT + r"\<vendor>\<grade>\<id>" "\n"
                             r"      \<time>_<shot>_<lighting>.jpg")
                        dpg.add_spacer(height=12)

                        section("Board IDs")
                        with dpg.group(horizontal=True):
                            dpg.add_input_int(label="Start ID", tag="start_id",
                                              default_value=147, width=140)
                            dpg.add_spacer(width=16)
                            dpg.add_input_int(label="How many", tag="total",
                                              default_value=5000, width=140)
                        dpg.add_button(label="Restart session at Start ID",
                                       callback=start_session, width=-1, height=34)
                        note("Not needed to begin - the first capture starts the "
                             "session. The saved Start ID is wherever you got to, so "
                             "a relaunch resumes instead of repeating a board.")
                        dpg.add_spacer(height=12)

                        section("Saved files")
                        dpg.add_checkbox(label="Legacy colour on save", tag="legacy_color",
                                         default_value=True)
                        note("On = byte-identical to the old script (R and B swapped "
                             "in the file). Off = the file matches the screen.")
                        dpg.add_spacer(height=12)

                        section("Appearance")
                        with dpg.group(horizontal=True):
                            dpg.add_combo(families, label="Font", tag="font_family",
                                          default_value="Default",
                                          width=150, callback=apply_look)
                            dpg.add_spacer(width=16)
                            dpg.add_combo(list(TEXT_SIZES), label="Size", tag="text_size",
                                          default_value="Normal", width=120,
                                          callback=apply_look)
                        dpg.add_button(label="Light / dark", callback=toggle_theme,
                                       width=-1, height=34)
                        note("Default is DearPyGui's built-in bitmap font - always "
                             "sharp, one size. The TTF families give the size presets "
                             "but need a scaled display to look right.")

                    # ---------------------------------- right: the Jetson
                    with dpg.child_window(width=-1, autosize_y=True):
                        section("Gateway")
                        dpg.add_input_text(label="URL", tag="url",
                                           default_value=JETSON_URL, width=-110)
                        dpg.add_input_text(label="API key", tag="key",
                                           default_value=API_KEY, width=-110)
                        note("The .ts.net address works from any machine while Funnel "
                             "is on. On the same LAN as the Jetson, "
                             "http://<jetson-ip>:8000 is faster.")
                        note(f"Typed once - saved to {CONF} when a test passes "
                             f"and again on exit.")
                        dpg.add_spacer(height=12)

                        section("Model")
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label="Task", tag="task",
                                               default_value="classification", width=190)
                            dpg.add_spacer(width=16)
                            dpg.add_checkbox(label="TTA", tag="tta")
                        dpg.add_input_text(label="Version", tag="model_version",
                                           default_value="", width=190,
                                           hint="blank = active model")
                        dpg.add_spacer(height=12)

                        section("Live AI")
                        dpg.add_input_float(label="Timeout (s)", tag="timeout",
                                            default_value=8.0, min_value=1.0,
                                            max_value=60.0, step=1.0, width=150)
                        dpg.add_checkbox(label="Draw boxes on the feed", tag="boxes",
                                         default_value=True)
                        note("The switch and the interval are on the Capture tab, "
                             "beside the video.")
                        dpg.add_spacer(height=12)

                        section("Connection")
                        dpg.add_button(label="Test connection", width=-1, height=36,
                                       callback=lambda: threading.Thread(
                                           target=test_server, daemon=True).start())
                        role(dpg.add_text("not tested yet", tag="api_status", wrap=460,
                                          color=DIM_C), "body")
                        with dpg.tree_node(label="Last raw response"):
                            role(dpg.add_text("", tag="pred_raw", wrap=460,
                                              color=DIM_C), "small")

            # ============================================== log
            with dpg.tab(label="   Log   "):
                section("Everything that happened")
                role(dpg.add_text("", tag="log"), "small")

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Q, callback=capture)

    S["dark_theme"], S["light_theme"] = make_theme(True), make_theme(False)
    dpg.bind_theme(S["dark_theme"])
    dpg.set_primary_window("root", True)
    dpg.set_viewport_resize_callback(fit_image)


def draw_boxes(preview):
    """Detector output onto the frame. Classification models return none."""
    p = S["pred"]
    if not isinstance(p, dict) or not dpg.get_value("boxes"):
        return
    for d in (p.get("detections") or [])[:20]:
        b = _box(d) if isinstance(d, dict) else None
        if b:
            c = conf_color(_conf(d))[::-1]           # RGB constant -> BGR frame
            cv2.rectangle(preview, b[:2], b[2:], c, 2)
            cv2.putText(preview, f"{_label(d)} {_conf(d):.2f}", (b[0], max(12, b[1] - 6)),
                        FONT, 0.45, c, 1, cv2.LINE_AA)


def show_pred():
    """Runs every loop, not per frame: a result must land even if the feed stalls.
    Cheap - the identity check means widgets only touch on a new response."""
    p = S["pred"]
    if p is None or p is S["drawn"]:
        return
    S["drawn"] = p

    raw = json.dumps(p, indent=1)[:2000]
    dpg.set_value("pred_raw", raw)
    dpg.set_value("pred_raw_live", raw)

    if isinstance(p, dict) and "error" in p:
        dpg.configure_item("pred_label", color=RED)
        dpg.set_value("pred_label", "AI ERROR - see Settings")
        for i in range(ROWS):
            dpg.hide_item(f"bar{i}")
        return

    ranked = _ranked(p)
    if ranked:
        top, conf = ranked[0]
        dpg.configure_item("pred_label", color=conf_color(conf))
        dpg.set_value("pred_label", f"{top}   {conf * 100:.1f}%")
    else:
        dpg.configure_item("pred_label", color=DIM_C)
        dpg.set_value("pred_label", "no prediction")

    for i in range(ROWS):
        if i < len(ranked):
            name, c = ranked[i]
            dpg.configure_item(f"bar{i}", show=True, overlay=f"{name}   {c * 100:.1f}%")
            dpg.set_value(f"bar{i}", min(1.0, c))
            # only the winner carries the confidence colour; the rest stay quiet
            dpg.bind_item_theme(f"bar{i}", S["bars"][conf_color(c) if i == 0
                                                     else (95, 95, 100)])
        else:
            dpg.hide_item(f"bar{i}")


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None
    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected - GUI runs in offline mode.")
        S["ph"], S["src_w"] = int(PREVIEW_W * 0.75), 1920
    else:
        cam = cam_list.GetByIndex(0)
        cam.Init()
        try:
            cam.OffsetX.SetValue(0)
            cam.OffsetY.SetValue(0)
            cam.Width.SetValue(cam.Width.GetMax())
            cam.Height.SetValue(cam.Height.GetMax())
        except PySpin.SpinnakerException as ex:
            print("Could not set full resolution:", ex)
        w, h = int(cam.Width.GetValue() / ZOOM), int(cam.Height.GetValue() / ZOOM)
        S["ph"], S["src_w"] = int(PREVIEW_W * h / w), w    # never upscale past the sensor
        cam.BeginAcquisition()

    S["aspect"] = S["ph"] / S["pw"]
    S["idle_sub"] = ("Offline mode - the UI works, capture does not."
                     if cam is None else "waiting for the first frame...")
    S["idle"] = placeholder("NO CAMERA DETECTED", S["idle_sub"])
    S["tex"] = np.zeros(S["pw"] * S["ph"] * 3, dtype=np.float32)
    set_gamma()

    dpi_aware()
    dpg.create_context()
    dpg.create_viewport(title="Gibson Capture", width=1360, height=860,
                        min_width=760, min_height=560)
    families = build_fonts()
    build_ui(families)
    dpg.setup_dearpygui()
    load_conf()
    apply_look()
    dpg.show_viewport()
    fit_image()

    for fn in (detect_loop, save_loop):
        threading.Thread(target=fn, daemon=True).start()
    log("Ready. Press Q to capture.")

    try:
        while dpg.is_dearpygui_running():
            # One snapshot per iteration: a resize swaps S["tex"] for a different
            # shape, and this pairing keeps the buffer and its dimensions together.
            tex, pw, ph = S["tex"], S["pw"], S["ph"]
            f = grab(cam) if cam is not None else None
            if f is not None:
                S["frame"] = f
                prev = cv2.resize(f, (pw, ph))
                S["preview"] = prev                         # BGR, what the server wants
                shown = cv2.LUT(prev, S["lut"])
            elif S["frame"] is None:
                shown = cv2.resize(S["idle"], (pw, ph))     # never an unexplained black box
            else:
                shown = None                                # dropped frame, keep the last one
            if shown is not None:
                draw_boxes(shown)
                draw_pred_overlay(shown)
                draw_hud(shown)
                tex[:] = shown[..., ::-1].ravel() * np.float32(1 / 255)   # BGR -> texture RGB

            show_pred()
            light = LIGHTS[S["img"]].replace("_", " ")
            if S["detect"]:
                dpg.set_value("now_light", "Live AI is running")
                dpg.set_value("now_step", "capture is locked until you switch it off")
            else:
                dpg.set_value("now_light", f"Set lights to {light}")
                dpg.set_value("now_step",
                              f"then press Q for shot {S['img']} of 3, board {S['id']}"
                              if S["session"] else
                              f"then press Q to start board {dpg.get_value('start_id')}")
            dpg.set_value("c_saved", str(S["saved"]))
            n = SAVE_Q.qsize()
            dpg.set_value("c_queue", str(n))
            dpg.configure_item("c_queue", color=AMBER if n else DIM_C)
            dpg.render_dearpygui_frame()
    finally:
        S["run"] = False
        save_conf()                 # before the context goes, values live in it
        dpg.destroy_context()
        if not SAVE_Q.empty():
            print(f"Flushing {SAVE_Q.qsize()} queued image(s)...")
            SAVE_Q.join()
        if cam is not None:
            for fn in (cam.EndAcquisition, cam.DeInit):
                try:
                    fn()
                except Exception:
                    pass
            del cam
        cam_list.Clear()
        system.ReleaseInstance()


def selftest():
    body, ct = _multipart([("task", "classification"), ("tta", "false")],
                          [("images", "a.jpg", b"\xff\xd8A"), ("images", "b.jpg", b"\xff\xd8B")])
    assert ct.split("boundary=")[1].encode() in body and body.endswith(b"--\r\n")
    assert body.count(b'name="images"') == 2 and b'name="task"\r\n\r\nclassification' in body
    # the server's own envelope
    env = {"label": "gradeA", "confidence": .44, "margin": .06,
           "probs": {"gradeA": .44, "gradeB": .19, "gradeC": .37}, "latency_ms": 944.1}
    assert _ranked(env) == [("gradeA", .44), ("gradeC", .37), ("gradeB", .19)]
    assert _ranked({"label": "x", "confidence": .5}) == [("x", .5)]
    assert _ranked({"detections": [{"class": "knot", "score": .8}]}) == [("knot", .8)]
    assert _ranked({"nope": 1}) == []
    assert conf_color(.9) == GREEN and conf_color(.6) == AMBER and conf_color(.2) == RED
    assert conf_color(.75) == GREEN and conf_color(.5) == AMBER      # boundaries
    for raw, want in (("100.103.105.68:8000", "http://100.103.105.68:8000"),
                      ("http://192.168.1.9:8000/", "http://192.168.1.9:8000"),
                      (" http://jetson:8000/ui ", "http://jetson:8000"),
                      ("http://jetson:8000/v1", "http://jetson:8000"),
                      ("drstreet.taildab2f8.ts.net", "https://drstreet.taildab2f8.ts.net")):
        assert _base(raw) == want, (raw, _base(raw))
    assert _tailnet("http://100.103.105.68:8000")              # CGNAT, tailnet only
    assert not _tailnet("https://drstreet.taildab2f8.ts.net")  # Funnel: public ingress
    assert not _tailnet("http://192.168.1.9:8000")
    assert not _tailnet("http://100.20.3.4:8000")              # public 100.x, not CGNAT
    assert "needs Tailscale" in _hint("timed out", "http://100.103.105.68:8000")
    assert "Funnel is still on" in _hint("timed out", "https://x.ts.net")
    assert "REFUSED" in _hint("ConnectionRefusedError: refused")
    assert "KEY REJECTED" in _hint("HTTP 401: invalid API key")
    assert _hint("boom") == "boom"
    set_gamma(g=1.0)
    assert S["lut"][0] == 0 and S["lut"][255] == 255 and S["lut"][128] == 128  # passthrough
    set_gamma(g=2.2)
    assert S["lut"][128] < 128 and S["lut"][255] == 255                        # darkens midtones
    # grab() gives BGR; the texture upload must swap it or faces go blue
    assert list(np.array([[[0, 0, 255]]], np.uint8)[..., ::-1].ravel()) == [255, 0, 0]
    print("selftest ok")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
