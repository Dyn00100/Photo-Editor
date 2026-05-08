import streamlit as st
import cv2
import numpy as np

def apply_white_balance(img):
    # Step 3: White Balance (Gray World Algorithm)
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

st.title("Duyen’s Pro Portrait Studio")

uploaded_file = st.file_uploader("Upload photo", type=["jpg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Step 1: Identify the Subject (Face)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Step 2: Smooth Face Only
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        # Smoothing just the detected rectangle
        blurred_face = cv2.bilateralFilter(face_roi, 9, 75, 75)
        img[y:y+h, x:x+w] = blurred_face

    # Step 3: Apply White Balance
    final_img = apply_white_balance(img)

    # Show result
    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Download button
    _, buffer = cv2.imencode('.png', final_img)
    st.download_button("Download Photo", data=buffer.tobytes(), file_name="edited.png")
