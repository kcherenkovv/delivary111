import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели
model = YOLO("model_11v_optimized_nz.onnx")  # замени на своё имя

# Приём стрима
cap = cv2.VideoCapture("udp://@0.0.0.0:5000", cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Ошибка открытия входящего потока")
    exit()

# Отправка обработанного видео
gst_out = (
    'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast '
    '! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.10 port=5001'
)
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, 25, (640, 480), True)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    annotated_frame = results[0].plot()  # Рисуем боксы

    # Изменение размера под оптимальную передачу
    resized_frame = cv2.resize(annotated_frame, (640, 480))

    out.write(resized_frame)

    # Для дебага можно раскомментировать:
    # cv2.imshow('Server Output', resized_frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
