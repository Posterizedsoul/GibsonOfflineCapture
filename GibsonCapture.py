import PySpin
import cv2
import os
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
JETSON_URL = "http://100.103.105.68:8000"          # Tailscale address, see ACCESS.md
API_KEY = os.environ.get("GIBSON_API_KEY", "")     # avoids retyping it every launch
ZOOM = 1.13              # calibration knob: lens/sensor crop, tune per rig
GAMMA = 1.0              # calibration knob: display only, 1.0 = raw passthrough
PREVIEW_W = 760          # texture width, keep small - these are old machines
PANEL_W = 340
ROWS = 6                 # class-probability bars in the detection tab
FONTS = (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf")
LIGHTS = {1: "Ring_and_Small_Lights", 2: "Only_Ring_Light", 3: "Only_Small_Lights"}

S = {
    "run": True, "frame": None, "preview": None,
    "id": 0, "img": 1, "last_id": -1, "session": False,
    "detect": False, "pred": None, "drawn": None, "log": [], "dark": True,
    "lut": None, "pw": PREVIEW_W, "ph": 0, "fonts": {},
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


# The Jetson is on a tailnet, i.e. a direct route. A machine with a corporate
# proxy configured would otherwise send 100.x traffic to the proxy and hang.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _base(u):
    """Tolerate what people actually paste: no scheme, trailing slash, /ui, /v1."""
    u = u.strip().rstrip("/")
    if not u.startswith(("http://", "https://")):
        # tailscale serve terminates TLS on 443, so a bare MagicDNS name is https
        u = ("https://" if u.endswith(".ts.net") else "http://") + u
    for tail in ("/ui", "/v1"):
        if u.endswith(tail):
            u = u[:-len(tail)]
    return u


def _tailnet(url):
    """A 100.64.0.0/10 address or a MagicDNS name - both need Tailscale on THIS pc.
    A .ts.net name resolves publicly but still points at the CGNAT address, so it
    is not a way around installing Tailscale; only Funnel would be."""
    return ".ts.net" in url or any(f"//100.{n}." in url for n in range(64, 128))


def _hint(err, url=""):
    """Turn a raw exception into the thing to go and check."""
    e = err.lower()
    if "timed out" in e or "timeout" in e:
        # The gateway publishes 8000 on the host, so a capture PC on the same
        # network as the Jetson reaches it directly, no Tailscale involved.
        return ("NO ROUTE to a tailnet address - Tailscale must be running on THIS pc "
                "(tray icon: Connected). No Tailscale here? Use the Jetson's LAN IP "
                "instead, http://192.168.x.x:8000" if _tailnet(url) else
                "NO ROUTE - nothing answered. Check the address and that the PC can "
                "reach that network.")
    if "refused" in e:
        return "REFUSED - address reached but nothing listening. Right port? Gateway running?"
    if "unknown url type" in e or "no host" in e:
        return "BAD URL - needs to look like http://100.103.105.68:8000"
    if "name or service" in e or "getaddrinfo" in e:
        return "CANNOT RESOLVE - use the 100.x address rather than a hostname."
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
    # Two attempts: over a relayed tailnet path the first TLS handshake regularly
    # times out while the second, on a warm path, is fine. Retrying only on
    # timeout keeps a real 401 or refusal instant.
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


def send_once(frame_bgr):
    return predict(dpg.get_value("url"), dpg.get_value("key"), frame_bgr,
                   dpg.get_value("task"), dpg.get_value("model_version"),
                   dpg.get_value("tta"), dpg.get_value("timeout"))


def detect_loop():
    """Live detection: latest preview -> POST /v1/predict -> result. Own thread."""
    while S["run"]:
        f = S["preview"]
        if not S["detect"] or f is None:
            time.sleep(0.15)
            continue
        t = time.time()
        S["pred"] = send_once(f)
        time.sleep(max(0.0, dpg.get_value("interval") - (time.time() - t)))


def test_server():
    url = _base(dpg.get_value("url"))
    set_status(f"testing {url} ...")
    h = health(url, max(10.0, dpg.get_value("timeout")))
    if "error" in h:
        set_status(f"{_hint(h['error'], url)}\n[{url}/health]  {h['error']}")
        return
    if not dpg.get_value("key"):
        set_status(f"server up ({json.dumps(h)[:60]}) - now paste an API key")
        return
    probe = S["preview"] if S["preview"] is not None else np.zeros((64, 64, 3), np.uint8)
    r = predict(url, dpg.get_value("key"), probe, dpg.get_value("task"),
                dpg.get_value("model_version"), False, dpg.get_value("timeout"))
    if "error" in r:
        set_status(f"server up, predict failed\n{_hint(r['error'], url)}\n{r['error']}")
        return
    S["pred"] = r
    set_status(f"OK - {r.get('model_id')}:{r.get('model_version')} "
               f"answered {r.get('label')} in {r.get('latency_ms', 0):.0f} ms")


def set_status(msg):
    dpg.set_value("api_status", msg)
    log(msg)


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
    S["log"] = (S["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-200:]
    dpg.set_value("log", "\n".join(S["log"][-40:]))


def start_session():
    S["id"] = dpg.get_value("start_id")
    S["last_id"] = S["id"] + dpg.get_value("total") - 1
    S["img"], S["session"] = 1, True
    log(f"Session started: IDs {S['id']}-{S['last_id']}")


def capture():
    """Queue the frame and return immediately - the feed keeps running."""
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
            return
        log(f"--> Next ID: {S['id']}")
    log(f"--> SWITCH LIGHTING TO: {LIGHTS[S['img']].replace('_', ' ')}")


# ------------------------------------------------------------------ look
def placeholder(msg, sub):
    """So the video area never sits there as an unexplained black rectangle."""
    img = np.full((S["ph"], S["pw"], 3), 28, np.uint8)
    cv2.putText(img, msg, (30, S["ph"] // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (210, 210, 210), 2)
    cv2.putText(img, sub, (30, S["ph"] // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 150, 150), 1)
    return img


def set_gamma(_=None, g=None):
    S["lut"] = (np.linspace(0, 1, 256) ** (GAMMA if g is None else g) * 255).astype(np.uint8)


def make_theme(dark):
    bg, child, frame, btn, hov, act, txt, dim = (
        ((20, 20, 22), (32, 32, 35), (50, 50, 54), (66, 66, 70), (98, 98, 104),
         (130, 130, 136), (245, 245, 245), (145, 145, 150))
        if dark else
        ((240, 240, 242), (255, 255, 255), (228, 228, 231), (218, 218, 222), (198, 198, 202),
         (170, 170, 175), (18, 18, 20), (110, 110, 115)))
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            for k, v in ((dpg.mvStyleVar_FrameRounding, 5), (dpg.mvStyleVar_ChildRounding, 7),
                         (dpg.mvStyleVar_WindowRounding, 7), (dpg.mvStyleVar_GrabRounding, 5),
                         (dpg.mvStyleVar_TabRounding, 5), (dpg.mvStyleVar_FrameBorderSize, 1)):
                dpg.add_theme_style(k, v, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7, category=dpg.mvThemeCat_Core)
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
                         (dpg.mvThemeCol_CheckMark, txt), (dpg.mvThemeCol_PlotHistogram, txt),
                         (dpg.mvThemeCol_ScrollbarBg, child), (dpg.mvThemeCol_ScrollbarGrab, frame),
                         (dpg.mvThemeCol_Separator, frame)):
                dpg.add_theme_color(k, v, category=dpg.mvThemeCat_Core)
    return t


def toggle_theme():
    S["dark"] = not S["dark"]
    dpg.bind_theme(S["dark_theme"] if S["dark"] else S["light_theme"])


def fit_image(*_):
    """Scale the preview to whatever room the window currently has."""
    w = dpg.get_viewport_client_width() - PANEL_W - 46
    h = dpg.get_viewport_client_height() - 245          # status line + log strip
    s = max(0.15, min(w / S["pw"], h / S["ph"]))
    dpg.configure_item("img", width=int(S["pw"] * s), height=int(S["ph"] * s))


def set_scale(_, v):
    dpg.set_global_font_scale(v)
    fit_image()


# ------------------------------------------------------------------ ui
def build_ui(tex):
    for p in FONTS:
        if os.path.exists(p):
            with dpg.font_registry():
                S["fonts"] = {n: dpg.add_font(p, n) for n in (17, 21, 32)}
            dpg.bind_font(S["fonts"][17])
            break

    with dpg.texture_registry():
        dpg.add_raw_texture(S["pw"], S["ph"], tex, format=dpg.mvFormat_Float_rgb, tag="tex")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            # ------------------------------------------------ live feed
            with dpg.child_window(width=-(PANEL_W + 8), autosize_y=True, border=False):
                dpg.add_image("tex", tag="img")
                dpg.add_text("No active session", tag="hud")
                with dpg.child_window(height=118):
                    dpg.add_text("", tag="log")

            # ------------------------------------------------ controls
            with dpg.child_window(width=PANEL_W, autosize_y=True, border=False):
                with dpg.child_window(height=-96):
                    with dpg.tab_bar():

                        with dpg.tab(label="Capture"):
                            dpg.add_input_text(label="Vendor", tag="vendor",
                                               default_value="Electric_Hardwood", width=130)
                            dpg.add_input_text(label="Grade", tag="grade",
                                               default_value="2A", width=130)
                            dpg.add_input_int(label="Start ID", tag="start_id",
                                              default_value=147, width=130)
                            dpg.add_input_int(label="Total IDs", tag="total",
                                              default_value=5000, width=130)
                            dpg.add_button(label="Start session", callback=start_session,
                                           width=-1, height=32)
                            dpg.add_separator()
                            dpg.add_text("Saving to")
                            dpg.add_text(SAVE_ROOT, tag="path_hint", wrap=290)

                        with dpg.tab(label="Detection"):
                            dpg.add_checkbox(label="Run live detection", tag="detect_cb",
                                             callback=lambda s, v: S.__setitem__("detect", v))
                            dpg.add_slider_float(label="Every s", tag="interval",
                                                 default_value=1.0, min_value=0.2,
                                                 max_value=10.0, width=110)
                            dpg.add_checkbox(label="Draw boxes", tag="boxes", default_value=True)
                            dpg.add_separator()
                            dpg.add_text("-", tag="pred_label")
                            dpg.add_text("", tag="pred_meta", wrap=300)
                            dpg.add_spacer(height=4)
                            for i in range(ROWS):
                                dpg.add_progress_bar(tag=f"bar{i}", width=-1,
                                                     overlay="", show=False)
                            with dpg.tree_node(label="Raw response"):
                                dpg.add_text("", tag="pred_raw", wrap=300)

                        with dpg.tab(label="Server"):
                            dpg.add_text("Jetson gateway")
                            dpg.add_input_text(tag="url", default_value=JETSON_URL, width=-1)
                            dpg.add_text("Same network as the Jetson: use its LAN IP. "
                                         "Off-site: the 100.x tailnet address.", wrap=300)
                            dpg.add_input_text(label="API key", tag="key",
                                               default_value=API_KEY, width=150)
                            dpg.add_input_text(label="Task", tag="task",
                                               default_value="classification", width=150)
                            dpg.add_input_text(label="Model ver", tag="model_version",
                                               default_value="", width=150,
                                               hint="blank = active model")
                            dpg.add_checkbox(label="TTA", tag="tta")
                            dpg.add_input_float(label="Timeout s", tag="timeout",
                                                default_value=8.0, min_value=1.0,
                                                max_value=60.0, step=1.0, width=110)
                            dpg.add_button(label="Test connection", width=-1, height=32,
                                           callback=lambda: threading.Thread(
                                               target=test_server, daemon=True).start())
                            dpg.add_text("", tag="api_status", wrap=300)

                        with dpg.tab(label="View"):
                            dpg.add_slider_float(label="Gamma", default_value=GAMMA,
                                                 min_value=0.4, max_value=3.0, width=110,
                                                 callback=lambda s, v: set_gamma(g=v))
                            dpg.add_slider_float(label="UI scale", default_value=1.0,
                                                 min_value=0.7, max_value=2.0, width=110,
                                                 callback=set_scale)
                            dpg.add_button(label="Light / dark", callback=toggle_theme, width=-1)
                            dpg.add_separator()
                            dpg.add_checkbox(label="Legacy colour on save",
                                             tag="legacy_color", default_value=True)
                            dpg.add_text("On = byte-identical to the old script "
                                         "(R and B swapped in the file).", wrap=300)

                # pinned: never hidden behind a tab
                dpg.add_button(label="CAPTURE   (Q)", callback=capture, width=-1, height=56)
                dpg.add_text("Queue: 0", tag="qstat")

    if S["fonts"]:
        dpg.bind_item_font("hud", S["fonts"][32])
        dpg.bind_item_font("pred_label", S["fonts"][21])

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Q, callback=capture)

    S["dark_theme"], S["light_theme"] = make_theme(True), make_theme(False)
    dpg.bind_theme(S["dark_theme"])
    dpg.set_primary_window("root", True)
    dpg.set_viewport_resize_callback(fit_image)


def draw_panel(preview):
    p = S["pred"]
    if p is None:
        return
    if dpg.get_value("boxes"):
        for d in (p.get("detections") or [])[:20]:
            b = _box(d) if isinstance(d, dict) else None
            if b:
                cv2.rectangle(preview, b[:2], b[2:], (255, 255, 255), 2)
                cv2.putText(preview, f"{_label(d)} {_conf(d):.2f}", (b[0], max(12, b[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if p is S["drawn"]:      # widgets only change when a new response lands
        return
    S["drawn"] = p
    if isinstance(p, dict) and "error" in p:
        dpg.set_value("pred_label", "ERROR")
        dpg.set_value("pred_meta", p["error"][:200])
        for i in range(ROWS):
            dpg.hide_item(f"bar{i}")
        return

    ranked = _ranked(p)
    dpg.set_value("pred_label", f"{ranked[0][0]}   {ranked[0][1] * 100:.1f}%"
                  if ranked else "no prediction")
    dpg.set_value("pred_meta",
                  f"{p.get('model_id', '?')}:{p.get('model_version', '?')}   "
                  f"{p.get('latency_ms', 0):.0f} ms"
                  + (f"   margin {p['margin']:.3f}" if p.get("margin") is not None else ""))
    for i in range(ROWS):
        if i < len(ranked):
            dpg.configure_item(f"bar{i}", show=True,
                               overlay=f"{ranked[i][0]}   {ranked[i][1] * 100:.1f}%")
            dpg.set_value(f"bar{i}", min(1.0, ranked[i][1]))
        else:
            dpg.hide_item(f"bar{i}")
    dpg.set_value("pred_raw", json.dumps(p)[:2000])


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None
    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected - GUI runs in offline mode.")
        S["ph"] = int(PREVIEW_W * 0.75)
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
        S["ph"] = int(PREVIEW_W * h / w)
        cam.BeginAcquisition()

    set_gamma()
    tex = np.zeros(S["pw"] * S["ph"] * 3, dtype=np.float32)

    dpg.create_context()
    dpg.create_viewport(title="Gibson Capture", width=1280, height=820,
                        min_width=700, min_height=520)
    build_ui(tex)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    fit_image()

    for fn in (detect_loop, save_loop):
        threading.Thread(target=fn, daemon=True).start()
    log("Ready. Q = capture.")

    idle = placeholder("NO CAMERA DETECTED",
                       "Offline mode - the UI works, capture does not."
                       if cam is None else "waiting for the first frame...")
    try:
        while dpg.is_dearpygui_running():
            f = grab(cam) if cam is not None else None
            if f is not None:
                S["frame"] = f
                prev = cv2.resize(f, (S["pw"], S["ph"]))
                S["preview"] = prev                         # BGR, what the server wants
                shown = cv2.LUT(prev, S["lut"])
            elif S["frame"] is None:
                shown = idle.copy()                         # never an unexplained black box
            else:
                shown = None                                # dropped frame, keep the last one
            if shown is not None:
                draw_panel(shown)
                tex[:] = shown[..., ::-1].ravel() * np.float32(1 / 255)   # BGR -> texture RGB
            dpg.set_value("hud", f"ID {S['id']}    {S['img']} of 3    "
                                 f"{LIGHTS[S['img']].replace('_', ' ')}"
                          if S["session"] else
                          f"Press Q to start at ID {dpg.get_value('start_id')}")
            dpg.set_value("qstat", f"Queue: {SAVE_Q.qsize()}")
            dpg.render_dearpygui_frame()
    finally:
        S["run"] = False
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
    for raw, want in (("100.103.105.68:8000", "http://100.103.105.68:8000"),
                      ("http://192.168.1.9:8000/", "http://192.168.1.9:8000"),
                      (" http://jetson:8000/ui ", "http://jetson:8000"),
                      ("http://jetson:8000/v1", "http://jetson:8000"),
                      ("drstreet.taildab2f8.ts.net", "https://drstreet.taildab2f8.ts.net")):
        assert _base(raw) == want, (raw, _base(raw))
    assert _tailnet("https://drstreet.taildab2f8.ts.net")      # MagicDNS -> CGNAT
    assert _tailnet("http://100.103.105.68:8000")
    assert not _tailnet("http://192.168.1.9:8000")
    assert not _tailnet("http://100.20.3.4:8000")              # public 100.x, not CGNAT
    assert "Tailscale must be running" in _hint("timed out", "http://100.103.105.68:8000")
    assert "Tailscale" not in _hint("timed out", "http://192.168.1.9:8000")
    assert "REFUSED" in _hint("ConnectionRefusedError: refused")
    assert "KEY REJECTED" in _hint("HTTP 401: invalid API key")
    assert _hint("boom") == "boom"
    assert _box({"x": 10, "y": 10, "width": 4, "height": 6}) == (8, 7, 12, 13)
    assert _box({"bbox": [1, 2, 3, 4]}) == (1, 2, 4, 6)
    assert _box({"x1": 1, "y1": 2, "x2": 3, "y2": 4}) == (1, 2, 3, 4)
    assert _box({}) is None
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
