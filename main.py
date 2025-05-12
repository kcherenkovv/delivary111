import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели
model = YOLO("model_11v_optimized_nz.onnx")  # замени на имя своей ONNX-модели

# Приём стрима через GStreamer
gst_pipeline = (
    'udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264 ! '
    'rtph264depay ! decodebin ! videoconvert ! appsink'
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Ошибка открытия входящего потока")
    exit()

# Отправка обратно через UDP
gst_out = (
    'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast '
    '! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.8 port=5001'
)
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, 25, (640, 480), True)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow('YOLOv11 Detection', annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
