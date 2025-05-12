import cv2
import numpy as np
import subprocess  # ✅ Добавь этот импорт
from ultralytics import YOLO

# Параметры видеопотока
WIDTH = 640
HEIGHT = 480
FFMPEG_CMD = [
    'ffmpeg',
    '-i', 'udp://@0.0.0.0:5000',
    '-pix_fmt', 'bgr24',      # Выходной формат OpenCV
    '-f', 'image2pipe',       # Передача кадров через pipe
    '-vcodec', 'rawvideo',     # Не декодировать, просто передать
    '-'
]

# Запуск ffmpeg как subprocess
process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Загрузка модели
model = YOLO("model_11v_optimized_nz.onnx")

while True:
    raw_frame = process.stdout.read(WIDTH * HEIGHT * 3)  # 3 байта на пиксель (BGR)
    if len(raw_frame) != WIDTH * HEIGHT * 3:
        print("Ошибка чтения кадра")
        continue

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

    # Детекция
    results = model(frame)
    annotated_frame = results[0].plot()

    # Показываем результат
    cv2.imshow('YOLOv11 Detection', annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Очистка
process.terminate()
cv2.destroyAllWindows()
