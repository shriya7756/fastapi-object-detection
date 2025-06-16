from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import os
import io
import threading
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

capture = None
record = False
out = None
frame_stats = {"total": 0, "confidence": 0, "fps": 0, "time": 0}
lock = threading.Lock()


def detect_objects(frame, thick_box=False):
    start_time = time.time()
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > 0.5:
                cx, cy, w, h = (det[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(cx - w / 2), int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    total_conf = 0
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]*100:.1f}%"
        total_conf += confidences[i]
        color = (0, 255, 0)
        thickness = 6 if thick_box else 4
        font_scale = 1.2 if thick_box else 0.9
        font_thick = 3 if thick_box else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick)

    elapsed = (time.time() - start_time) * 1000
    with lock:
        frame_stats["total"] = len(indexes)
        frame_stats["confidence"] = int((total_conf / max(len(indexes), 1)) * 100)
        frame_stats["time"] = int(elapsed)

    return frame


def generate_frames():
    global capture, record, out
    capture = cv2.VideoCapture(0)
    prev_time = time.time()

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        frame = detect_objects(frame)
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time + 1e-6))
        prev_time = cur_time

        with lock:
            frame_stats["fps"] = fps

        if record and out is not None:
            out.write(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    capture.release()
    if out:
        out.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = detect_objects(img, thick_box=True)
    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.get("/stats/")
def stats():
    with lock:
        return JSONResponse(frame_stats)


@app.post("/start_camera")
def start_camera():
    global capture
    capture = cv2.VideoCapture(0)
    return {"status": "camera started"}


@app.post("/stop_camera")
def stop_camera():
    global capture
    if capture and capture.isOpened():
        capture.release()
    return {"status": "camera stopped"}


@app.post("/start_record")
def start_recording():
    global record, out, capture
    if capture is None or not capture.isOpened():
        capture = cv2.VideoCapture(0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("recorded_output.avi", cv2.VideoWriter_fourcc(*"XVID"), 20, (width, height))
    record = True
    return {"status": "recording started"}


@app.post("/stop_record")
def stop_recording():
    global record, out
    record = False
    if out:
        out.release()
    return {"status": "recording stopped"}


@app.get("/download_recording")
def download_recording():
    file_path = "recorded_output.avi"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="recorded_output.avi", media_type='video/x-msvideo')
    return JSONResponse({"error": "Recording not found"}, status_code=404)
