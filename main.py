from ultralytics import YOLO
import cv2

# Путь к твоей модели в формате .onnx
model_path = "model_11v_optimized_nz.onnx"  # замени на свой путь

# Загрузка модели (Ultralytics автоматически определит, что это ONNX)
model = YOLO(model_path)

# Открытие видеокамеры (обычно индекс 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

print("Запуск камеры... Нажмите 'q' чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр.")
        break

    # Детекция на CPU (Ultralytics сама использует onnxruntime при работе с .onnx)
    results = model(frame)

    # Рисование результатов на кадре
    annotated_frame = results[0].plot()

    # Отображение результата
    cv2.imshow('YOLOv Detection - ONNX CPU', annotated_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
