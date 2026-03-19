import os
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="PulmoScan AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.environ.get("MODEL_PATH", "trained_lung_cancer_model.h5")
IMAGE_SIZE = (350, 350)

DEFAULT_CLASS_LABELS = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
]

raw_labels = os.environ.get("CLASS_LABELS", "")
class_labels = [item.strip() for item in raw_labels.split(",") if item.strip()] if raw_labels.strip() else DEFAULT_CLASS_LABELS


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)


st.title("🫁 PulmoScan AI")
st.subheader("Lung Cancer Detection using Deep Learning")
st.write("---")

st.markdown("""
Upload a CT scan image to get an AI-powered prediction of lung cancer classification.
""")

model = load_trained_model()

uploaded_file = st.file_uploader("📁 Choose a CT scan image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Analysis")
        
        image_resized = image.resize(IMAGE_SIZE)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = class_labels[pred_index] if pred_index < len(class_labels) else str(pred_index)
        
        color = "🟢" if predicted_label == "Normal" else "🔴"
        st.markdown(f"### {color} {predicted_label}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        
        st.write("---")
        st.subheader("All Class Probabilities")
        
        for i, label in enumerate(class_labels):
            prob = float(predictions[0][i])
            st.write(f"**{label}**: {prob * 100:.2f}%")
            st.progress(prob)

st.write("---")
st.markdown("""
### ⚠️ Medical Disclaimer
This AI analysis is for screening purposes only and **should not be used as a final diagnosis**. 
Always consult with a qualified healthcare professional for medical advice and diagnosis.
""")

