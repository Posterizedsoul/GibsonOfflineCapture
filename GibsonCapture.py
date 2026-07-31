import PySpin
import cv2
import os
import time
import json
import uuid
import threading
import urllib.request
import numpy as np
import dearpygui.dearpygui as dpg

# ==========================================
# CONFIGURATION (editable in the GUI at runtime)
# ==========================================
SAVE_ROOT = r"C:\Gibson"
JETSON_URL = "http://jetson.local:8000/predict"
ZOOM = 1.13              # calibration knob: lens/sensor crop, tune per rig
PREVIEW_W = 760          # keep small, these are old machines
LIGHTS = {1: "Ring_and_Small_Lights", 2: "Only_Ring_Light", 3: "Only_Small_Lights"}

S = {
    "run": True, "frame": None, "preview": None, "cam": None,
    "id": 0, "img": 1, "last_id": -1, "session": False,
    "detect": False, "pred": None, "ms": 0, "log": [],
}


# ------------------------------------------------------------------ Jetson API
def _multipart(jpg, field="file", name="frame.jpg"):
    """Build a multipart/form-data body without pulling in `requests`."""
    b = uuid.uuid4().hex
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{name}\"\r\nContent-Type: image/jpeg\r\n\r\n"
    ).encode() + jpg + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def post_frame(url, bgr, timeout=4.0):
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    body, ctype = _multipart(jpg.tobytes())
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


def detect_loop():
    """Posts the *preview* frame so box coords land in preview space directly."""
    while S["run"]:
        f = S["preview"]
        if not S["detect"] or f is None:
            time.sleep(0.15)
            continue
        t = time.time()
        try:
            S["pred"] = post_frame(dpg.get_value("url"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        except Exception as ex:
            S["pred"] = {"error": f"{type(ex).__name__}: {ex}"}
        S["ms"] = int((time.time() - t) * 1000)
        time.sleep(max(0.0, dpg.get_value("interval") - (time.time() - t)))


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
    if not S["session"] or S["frame"] is None:
        log("Nothing to capture (start a session first)")
        return
    folder = os.path.join(SAVE_ROOT, dpg.get_value("vendor"), dpg.get_value("grade"), str(S["id"]))
    os.makedirs(folder, exist_ok=True)
    name = f"{time.strftime('%H-%M-%S')}_{S['img']}_{LIGHTS[S['img']]}.jpg"
    cv2.imwrite(os.path.join(folder, name), cv2.cvtColor(S["frame"], cv2.COLOR_RGB2BGR))
    log(f"Saved {S['id']}/{name}")

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


# ------------------------------------------------------------------ ui
def theme():
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            for k, v in ((dpg.mvStyleVar_FrameRounding, 6), (dpg.mvStyleVar_ChildRounding, 8),
                         (dpg.mvStyleVar_WindowRounding, 8), (dpg.mvStyleVar_GrabRounding, 6)):
                dpg.add_theme_style(k, v, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7, category=dpg.mvThemeCat_Core)
            for k, v in ((dpg.mvThemeCol_WindowBg, (22, 24, 28)),
                         (dpg.mvThemeCol_ChildBg, (28, 31, 36)),
                         (dpg.mvThemeCol_FrameBg, (38, 42, 49)),
                         (dpg.mvThemeCol_Button, (0, 122, 204)),
                         (dpg.mvThemeCol_ButtonHovered, (26, 148, 230)),
                         (dpg.mvThemeCol_ButtonActive, (0, 96, 164)),
                         (dpg.mvThemeCol_Header, (0, 122, 204)),
                         (dpg.mvThemeCol_PlotHistogram, (46, 204, 113))):
                dpg.add_theme_color(k, v, category=dpg.mvThemeCat_Core)
    return t


def build_ui(pw, ph, tex):
    dpg.add_texture_registry(tag="tr")
    dpg.add_raw_texture(pw, ph, tex, format=dpg.mvFormat_Float_rgb, tag="tex", parent="tr")

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=pw + 16, autosize_y=True):
                dpg.add_image("tex")
                dpg.add_text("", tag="hud", color=(46, 204, 113))
                dpg.add_text("", tag="log")

            with dpg.child_window(width=360, autosize_y=True):
                dpg.add_text("SESSION", color=(120, 170, 255))
                dpg.add_input_text(label="Vendor", tag="vendor", default_value="Electric_Hardwood", width=180)
                dpg.add_input_text(label="Grade", tag="grade", default_value="2A", width=180)
                dpg.add_input_int(label="Start ID", tag="start_id", default_value=147, width=180)
                dpg.add_input_int(label="Total IDs", tag="total", default_value=5000, width=180)
                dpg.add_button(label="Start session", callback=start_session, width=-1, height=34)
                dpg.add_separator()
                dpg.add_button(label="CAPTURE  (Q)", callback=capture, width=-1, height=48)
                dpg.add_separator()

                dpg.add_text("JETSON AI", color=(120, 170, 255))
                dpg.add_input_text(label="", tag="url", default_value=JETSON_URL, width=-1)
                dpg.add_checkbox(label="Live detection", tag="detect_cb",
                                 callback=lambda s, v: S.__setitem__("detect", v))
                dpg.add_slider_float(label="Interval s", tag="interval", default_value=1.0,
                                     min_value=0.1, max_value=5.0, width=140)
                dpg.add_checkbox(label="Draw boxes", tag="boxes", default_value=True)
                dpg.add_text("idle", tag="pred_top", color=(46, 204, 113))
                dpg.add_progress_bar(tag="pred_bar", default_value=0.0, width=-1)
                dpg.add_text("", tag="pred_meta")
                with dpg.collapsing_header(label="Raw response"):
                    dpg.add_text("", tag="pred_raw", wrap=330)

    dpg.bind_theme(theme())
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Q, callback=capture)
    dpg.set_primary_window("root", True)


def draw_panel(preview):
    p = S["pred"]
    if p is None:
        return
    if isinstance(p, dict) and "error" in p:
        dpg.set_value("pred_top", "API ERROR")
        dpg.set_value("pred_bar", 0.0)
        dpg.set_value("pred_meta", p["error"][:120])
        dpg.set_value("pred_raw", "")
        return

    dets = sorted(_preds(p), key=_conf, reverse=True)
    if not dets:
        dpg.set_value("pred_top", "no detections")
        dpg.set_value("pred_bar", 0.0)
    else:
        dpg.set_value("pred_top", f"{_label(dets[0])}   {_conf(dets[0]) * 100:.1f}%")
        dpg.set_value("pred_bar", min(1.0, _conf(dets[0])))
    dpg.set_value("pred_meta", f"{len(dets)} object(s)  |  {S['ms']} ms")
    dpg.set_value("pred_raw", json.dumps(p)[:1500])

    if dpg.get_value("boxes"):
        for d in dets[:20]:
            b = _box(d) if isinstance(d, dict) else None
            if b:
                cv2.rectangle(preview, b[:2], b[2:], (46, 204, 113), 2)
                cv2.putText(preview, f"{_label(d)} {_conf(d):.2f}", (b[0], max(12, b[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1)


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None
    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected - GUI runs in offline mode.")
        pw, ph = PREVIEW_W, int(PREVIEW_W * 0.75)
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
        pw, ph = PREVIEW_W, int(PREVIEW_W * h / w)
        cam.BeginAcquisition()

    tex = np.zeros(pw * ph * 3, dtype=np.float32)

    dpg.create_context()
    dpg.create_viewport(title="Gibson Capture", width=pw + 420, height=ph + 260)
    build_ui(pw, ph, tex)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    threading.Thread(target=detect_loop, daemon=True).start()
    log("Ready. Q = capture.")

    try:
        while dpg.is_dearpygui_running():
            if cam is not None:
                f = grab(cam)
                if f is not None:
                    S["frame"] = f
                    prev = cv2.resize(f, (pw, ph))
                    S["preview"] = prev
                    shown = prev.copy()
                    draw_panel(shown)
                    tex[:] = shown.ravel() * np.float32(1 / 255)
            else:
                draw_panel(np.zeros((ph, pw, 3), np.uint8))
            dpg.set_value("hud", f"ID {S['id']}  |  Image {S['img']}/3  |  "
                                 f"{LIGHTS[S['img']].replace('_', ' ')}"
                          if S["session"] else "No active session")
            dpg.render_dearpygui_frame()
    finally:
        S["run"] = False
        dpg.destroy_context()
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
    body, ct = _multipart(b"\xff\xd8jpg")
    assert b'name="file"' in body and body.endswith(b"--\r\n") and ct.startswith("multipart/")
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
    print("selftest ok")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
