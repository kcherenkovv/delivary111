import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Загрузка модели
model = YOLO("model_11v_optimized_nz.onnx")  # Используйте свою модель

# Интерфейс Streamlit
st.title("YOLO11 Live Inference (Cloud GPU)")
img_buffer = st.camera_input("Включите веб-камеру")

if img_buffer:
    # Преобразование кадра в numpy array
    bytes_data = img_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Детекция на GPU
    results = model(img, device="cuda")  # Используем GPU
    
    # Визуализация
    res_plotted = results[0].plot()  # Рамки и метки
    st.image(res_plotted, channels="BGR")
