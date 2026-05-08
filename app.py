import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Duyen’s Portrait Studio ✨")
st.write("Upload a portrait to apply a soft, warm professional edit.")

# 1. The Upload Button
uploaded_file = st.file_uploader("Choose a portrait photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 2. The "Portrait Magic" Logic
    # Soften skin (Bilateral Filter)
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Add Warmth
    # Creating a peach/orange tint
    warm_layer = np.full(smoothed.shape, (150, 180, 230), dtype=np.uint8) 
    final_img = cv2.addWeighted(smoothed, 0.85, warm_layer, 0.15, 0)

    # 3. Show the Results
    # Convert BGR back to RGB for the website to show it correctly
    display_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    st.image(display_img, caption="Your Professional Portrait", use_container_width=True)

    # 4. The Download Button
    _, buffer = cv2.imencode('.png', final_img)
    st.download_button(
        label="Download Edited Photo",
        data=buffer.tobytes(),
        file_name="duyen_portrait.png",
        mime="image/png"
    )
