import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "pneumonia_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1L2A9-p10IG2aIqLkn-wg-Q9NR-ZgdSJd"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- STREAMLIT APP ----------------
st.title("Pneumonia Detection from X-ray")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction * 100

    if prediction > 0.5:
        st.markdown("🟥 **PNEUMONIA DETECTED**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.write("**Suggestions:**")
        st.write("- Consult a doctor immediately")
        st.write("- Take prescribed antibiotics only")
        st.write("- Get adequate rest and fluids")
        st.write("- Monitor breathing and oxygen level")
    else:
        st.markdown("🟩 **NORMAL**")
        st.markdown(f"**Confidence:** {(100 - confidence):.2f}%")
        st.write("**Basic Health Care Tips:**")
        st.write("- Maintain hygiene")
        st.write("- Avoid smoking")
        st.write("- Do regular breathing exercises")
        st.write("- Eat healthy food")
