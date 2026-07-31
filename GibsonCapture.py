import PySpin
import cv2
import os
import time
import json
import uuid
import queue
import threading
import urllib.request
import numpy as np
import dearpygui.dearpygui as dpg

# ==========================================
# CONFIGURATION (most of it editable in the GUI at runtime)
# ==========================================
SAVE_ROOT = r"C:\Gibson"
JETSON_URL = "http://jetson.local:8000/predict"
ZOOM = 1.13              # calibration knob: lens/sensor crop, tune per rig
GAMMA = 2.2              # calibration knob: display only, 1.0 = raw passthrough
PREVIEW_W = 760          # texture width, keep small - these are old machines
PANEL_W = 340
FONTS = (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf")
LIGHTS = {1: "Ring_and_Small_Lights", 2: "Only_Ring_Light", 3: "Only_Small_Lights"}

S = {
    "run": True, "frame": None, "preview": None,
    "id": 0, "img": 1, "last_id": -1, "session": False,
    "detect": False, "pred": None, "ms": 0, "log": [], "dark": True,
    "lut": None, "pw": PREVIEW_W, "ph": 0,
}
SAVE_Q = queue.Queue()   # capture never blocks the live feed


# ------------------------------------------------------------------ Jetson API
def _multipart(jpg, field="file", name="frame.jpg"):
    """Build a multipart/form-data body without pulling in `requests`."""
    b = uuid.uuid4().hex
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{name}\"\r\nContent-Type: image/jpeg\r\n\r\n"
    ).encode() + jpg + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def post_frame(url, bgr, field="file", timeout=4.0):
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    body, ctype = _multipart(jpg.tobytes(), field)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _preds(resp):
    """Normalise the common response shapes into a list of detections."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in ("predictions", "detections", "results", "objects"):
            if isinstance(resp.get(k), list):
                return resp[k]
        if any(k in resp for k in ("class", "label", "name")):
            return [resp]
    return []


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


def send_once(frame_rgb):
    """One round trip. Returns the parsed response or an {'error': ...} dict."""
    try:
        return post_frame(dpg.get_value("url"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          dpg.get_value("field"), dpg.get_value("timeout"))
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def detect_loop():
    """Live detection: grab latest preview -> POST -> result. Own thread."""
    while S["run"]:
        f = S["preview"]
        if not S["detect"] or f is None:
            time.sleep(0.15)
            continue
        t = time.time()
        S["pred"] = send_once(f)
        S["ms"] = int((time.time() - t) * 1000)
        time.sleep(max(0.0, dpg.get_value("interval") - (time.time() - t)))


# ------------------------------------------------------------------ capture queue
def save_loop():
    """Disk writes happen here so the live feed never stalls on a capture."""
    while S["run"]:
        try:
            path, rgb = SAVE_Q.get(timeout=0.3)
        except queue.Empty:
            continue
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            log(f"saved {os.path.basename(path)}")
        except Exception as ex:
            log(f"SAVE FAILED: {ex}")
        finally:
            SAVE_Q.task_done()


# ------------------------------------------------------------------ camera
def grab(cam):
    try:
        img = cam.GetNextImage(1000)
    except PySpin.SpinnakerException:
        return None
    if img.IsIncomplete():
        img.Release()
        return None
    bayer = np.frombuffer(img.GetData(), dtype=np.uint8).reshape(img.GetHeight(), img.GetWidth())
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)  # copies, safe to Release after
    img.Release()
    h, w = rgb.shape[:2]
    nw, nh = int(w / ZOOM), int(h / ZOOM)
    return rgb[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw]


# ------------------------------------------------------------------ session
def log(msg):
    S["log"] = (S["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-14:]
    dpg.set_value("log", "\n".join(S["log"]))


def start_session():
    S["id"] = dpg.get_value("start_id")
    S["last_id"] = S["id"] + dpg.get_value("total") - 1
    S["img"], S["session"] = 1, True
    log(f"Session started: IDs {S['id']}-{S['last_id']}")


def capture():
    """Queue the frame and return immediately - the feed keeps running."""
    if not S["session"] or S["frame"] is None:
        log("Nothing to capture (start a session first)")
        return
    folder = os.path.join(SAVE_ROOT, dpg.get_value("vendor"), dpg.get_value("grade"), str(S["id"]))
    name = f"{time.strftime('%H-%M-%S')}_{S['img']}_{LIGHTS[S['img']]}.jpg"
    SAVE_Q.put((os.path.join(folder, name), S["frame"]))
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
def set_gamma(_=None, g=None):
    g = GAMMA if g is None else g
    S["lut"] = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)


def make_theme(dark):
    bg, child, frame, btn, hov, act, txt, dim = (
        ((18, 18, 18), (30, 30, 30), (48, 48, 48), (64, 64, 64), (96, 96, 96),
         (128, 128, 128), (245, 245, 245), (150, 150, 150))
        if dark else
        ((242, 242, 242), (255, 255, 255), (226, 226, 226), (216, 216, 216), (196, 196, 196),
         (168, 168, 168), (16, 16, 16), (110, 110, 110)))
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            for k, v in ((dpg.mvStyleVar_FrameRounding, 5), (dpg.mvStyleVar_ChildRounding, 7),
                         (dpg.mvStyleVar_WindowRounding, 7), (dpg.mvStyleVar_GrabRounding, 5),
                         (dpg.mvStyleVar_FrameBorderSize, 1)):
                dpg.add_theme_style(k, v, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7, category=dpg.mvThemeCat_Core)
            for k, v in ((dpg.mvThemeCol_WindowBg, bg), (dpg.mvThemeCol_ChildBg, child),
                         (dpg.mvThemeCol_PopupBg, child), (dpg.mvThemeCol_MenuBarBg, child),
                         (dpg.mvThemeCol_FrameBg, frame), (dpg.mvThemeCol_FrameBgHovered, hov),
                         (dpg.mvThemeCol_FrameBgActive, act), (dpg.mvThemeCol_Button, btn),
                         (dpg.mvThemeCol_ButtonHovered, hov), (dpg.mvThemeCol_ButtonActive, act),
                         (dpg.mvThemeCol_Header, btn), (dpg.mvThemeCol_HeaderHovered, hov),
                         (dpg.mvThemeCol_HeaderActive, act), (dpg.mvThemeCol_Border, frame),
                         (dpg.mvThemeCol_Text, txt), (dpg.mvThemeCol_TextDisabled, dim),
                         (dpg.mvThemeCol_TitleBg, child), (dpg.mvThemeCol_TitleBgActive, frame),
                         (dpg.mvThemeCol_SliderGrab, txt), (dpg.mvThemeCol_SliderGrabActive, txt),
                         (dpg.mvThemeCol_CheckMark, txt), (dpg.mvThemeCol_PlotHistogram, txt),
                         (dpg.mvThemeCol_ScrollbarBg, child), (dpg.mvThemeCol_ScrollbarGrab, frame)):
                dpg.add_theme_color(k, v, category=dpg.mvThemeCat_Core)
    return t


def toggle_theme():
    S["dark"] = not S["dark"]
    dpg.bind_theme(S["dark_theme"] if S["dark"] else S["light_theme"])


def fit_image(*_):
    """Scale the preview to whatever room the window currently has."""
    w = dpg.get_viewport_client_width() - PANEL_W - 46
    h = dpg.get_viewport_client_height() - 190
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
                dpg.bind_font(dpg.add_font(p, 18))
            break

    with dpg.texture_registry(tag="tr"):
        dpg.add_raw_texture(S["pw"], S["ph"], tex, format=dpg.mvFormat_Float_rgb, tag="tex")

    # --- API settings: its own window, opened from the menu
    with dpg.window(label="API Settings", tag="api_win", show=False, pos=(120, 120),
                    width=430, height=210, no_collapse=True):
        dpg.add_text("Jetson endpoint")
        dpg.add_input_text(tag="url", default_value=JETSON_URL, width=-1)
        dpg.add_input_text(label="Form field", tag="field", default_value="file", width=140)
        dpg.add_input_float(label="Timeout s", tag="timeout", default_value=4.0,
                            min_value=0.5, max_value=30.0, step=0.5, width=140)
        dpg.add_button(label="Test connection", width=-1, height=32, callback=lambda: threading.Thread(
            target=lambda: S.__setitem__("pred", send_once(S["preview"]))
            if S["preview"] is not None else log("no frame yet"), daemon=True).start())
        dpg.add_text("", tag="api_status", wrap=400)

    # --- live detection: its own window too, so it can sit on a second screen
    with dpg.window(label="Live Detection", tag="det_win", pos=(160, 380),
                    width=430, height=330, no_collapse=True):
        dpg.add_checkbox(label="Run live detection", tag="detect_cb",
                         callback=lambda s, v: S.__setitem__("detect", v))
        dpg.add_slider_float(label="Interval s", tag="interval", default_value=1.0,
                             min_value=0.1, max_value=5.0, width=150)
        dpg.add_checkbox(label="Draw boxes on feed", tag="boxes", default_value=True)
        dpg.add_separator()
        dpg.add_text("idle", tag="pred_top")
        dpg.add_progress_bar(tag="pred_bar", default_value=0.0, width=-1)
        dpg.add_text("", tag="pred_meta")
        with dpg.collapsing_header(label="Raw response"):
            dpg.add_text("", tag="pred_raw", wrap=400)

    with dpg.window(tag="root"):
        with dpg.menu_bar():
            with dpg.menu(label="Windows"):
                dpg.add_menu_item(label="API Settings", callback=lambda: dpg.show_item("api_win"))
                dpg.add_menu_item(label="Live Detection", callback=lambda: dpg.show_item("det_win"))
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Light / dark", callback=toggle_theme)
                dpg.add_slider_float(label="UI scale", default_value=1.0, min_value=0.7,
                                     max_value=2.0, width=140, callback=set_scale)
                dpg.add_slider_float(label="Gamma", default_value=GAMMA, min_value=1.0,
                                     max_value=3.0, width=140,
                                     callback=lambda s, v: set_gamma(g=v))

        with dpg.group(horizontal=True):
            with dpg.child_window(width=-(PANEL_W + 8), autosize_y=True):
                dpg.add_image("tex", tag="img")
                dpg.add_text("", tag="hud")
                dpg.add_text("", tag="log")

            with dpg.child_window(width=PANEL_W, autosize_y=True):
                dpg.add_text("SESSION")
                dpg.add_input_text(label="Vendor", tag="vendor",
                                   default_value="Electric_Hardwood", width=150)
                dpg.add_input_text(label="Grade", tag="grade", default_value="2A", width=150)
                dpg.add_input_int(label="Start ID", tag="start_id", default_value=147, width=150)
                dpg.add_input_int(label="Total IDs", tag="total", default_value=5000, width=150)
                dpg.add_button(label="Start session", callback=start_session, width=-1, height=34)
                dpg.add_separator()
                dpg.add_button(label="CAPTURE  (Q)", callback=capture, width=-1, height=54)
                dpg.add_text("", tag="qstat")

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
    if isinstance(p, dict) and "error" in p:
        dpg.set_value("pred_top", "API ERROR")
        dpg.set_value("pred_bar", 0.0)
        dpg.set_value("pred_meta", p["error"][:160])
        dpg.set_value("api_status", p["error"][:160])
        return

    dets = sorted(_preds(p), key=_conf, reverse=True)
    dpg.set_value("pred_top", f"{_label(dets[0])}   {_conf(dets[0]) * 100:.1f}%"
                  if dets else "no detections")
    dpg.set_value("pred_bar", min(1.0, _conf(dets[0])) if dets else 0.0)
    dpg.set_value("pred_meta", f"{len(dets)} object(s)  |  {S['ms']} ms")
    dpg.set_value("pred_raw", json.dumps(p)[:1500])
    dpg.set_value("api_status", "OK")

    if dpg.get_value("boxes"):
        for d in dets[:20]:
            b = _box(d) if isinstance(d, dict) else None
            if b:
                cv2.rectangle(preview, b[:2], b[2:], (255, 255, 255), 2)
                cv2.putText(preview, f"{_label(d)} {_conf(d):.2f}", (b[0], max(12, b[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


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
    dpg.create_viewport(title="Gibson Capture", width=1280, height=820, min_width=640, min_height=480)
    build_ui(tex)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    fit_image()

    for fn in (detect_loop, save_loop):
        threading.Thread(target=fn, daemon=True).start()
    log("Ready. Q = capture.")

    try:
        while dpg.is_dearpygui_running():
            if cam is not None:
                f = grab(cam)
                if f is not None:
                    S["frame"] = f
                    prev = cv2.resize(f, (S["pw"], S["ph"]))
                    S["preview"] = prev
                    shown = cv2.LUT(prev, S["lut"])
                    draw_panel(shown)
                    tex[:] = shown.ravel() * np.float32(1 / 255)
            else:
                draw_panel(np.zeros((S["ph"], S["pw"], 3), np.uint8))
            dpg.set_value("hud", f"ID {S['id']}   Image {S['img']}/3   "
                                 f"{LIGHTS[S['img']].replace('_', ' ')}"
                          if S["session"] else "No active session")
            dpg.set_value("qstat", f"Pending writes: {SAVE_Q.qsize()}")
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
    body, ct = _multipart(b"\xff\xd8jpg", "image")
    assert b'name="image"' in body and body.endswith(b"--\r\n") and ct.startswith("multipart/")
    assert ct.split("boundary=")[1].encode() in body
    assert _preds({"predictions": [{"class": "knot", "confidence": .9}]})[0]["class"] == "knot"
    assert _preds({"detections": [1, 2]}) == [1, 2]
    assert _preds({"class": "2A", "score": .5}) == [{"class": "2A", "score": .5}]
    assert _preds({"nope": 1}) == [] and _preds([{"a": 1}]) == [{"a": 1}]
    assert _conf({"score": .5}) == .5 and _label({"name": "x"}) == "x"
    assert _box({"x": 10, "y": 10, "width": 4, "height": 6}) == (8, 7, 12, 13)
    assert _box({"bbox": [1, 2, 3, 4]}) == (1, 2, 4, 6)
    assert _box({"x1": 1, "y1": 2, "x2": 3, "y2": 4}) == (1, 2, 3, 4)
    assert _box({}) is None
    set_gamma(g=1.0)
    assert S["lut"][0] == 0 and S["lut"][255] == 255 and S["lut"][128] == 128  # 1.0 = passthrough
    set_gamma(g=2.2)
    assert S["lut"][128] < 128 and S["lut"][255] == 255                        # darkens midtones
    print("selftest ok")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
