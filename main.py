from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

# Path to the templates folder
TEMPLATES_DIR = "templates"

# Load YOLO model
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get YOLO layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten().astype(int).tolist()]

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Video stream generator
def video_stream():
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO Object Detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Draw bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/")
def home():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
