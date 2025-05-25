import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# Параметры видеопотока
WIDTH = 640
HEIGHT = 480

# Команда для получения UDP-потока через FFmpeg
FFMPEG_RECEIVE_CMD = [
    'ffmpeg',
    '-i', 'udp://@0.0.0.0:5000',
    '-f', 'image2pipe',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-'
]
process_in = subprocess.Popen(FFMPEG_RECEIVE_CMD, stdout=subprocess.PIPE)

# Команда для отправки обратно через UDP
FFMPEG_SEND_CMD = [
    'ffmpeg',
    '-y',
    '-f', 'image2pipe',
    '-vcodec', 'mjpeg',
    '-r', '25',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-pix_fmt', 'yuvj420p',
    '-i', '-',
    '-f', 'mpegts',
    'udp://192.168.1.8:5500'
]
process_out = subprocess.Popen(FFMPEG_SEND_CMD, stdin=subprocess.PIPE)

# Загрузка модели
model = YOLO("model_11v_optimized_nz.onnx")

print("✅ Ожидание кадров...")

while True:
    raw_frame = process_in.stdout.read(WIDTH * HEIGHT * 3)
    if len(raw_frame) != WIDTH * HEIGHT * 3:
        print(f"⚠️ Ошибка чтения кадра: {len(raw_frame)} байт")
        continue

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

    # Детекция
    results = model(frame)
    annotated_frame = results[0].plot()

    # Отправляем обратно
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    process_out.stdin.write(buffer.tobytes())
