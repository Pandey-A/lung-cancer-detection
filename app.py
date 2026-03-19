import os
from io import BytesIO
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model


# Model and preprocessing configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "trained_lung_cancer_model.h5")
IMAGE_SIZE = (350, 350)

# Keep class order aligned with training generator class_indices
DEFAULT_CLASS_LABELS = [
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "normal",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
]

raw_labels = os.environ.get("CLASS_LABELS", "")
if raw_labels.strip():
    class_labels: List[str] = [item.strip() for item in raw_labels.split(",") if item.strip()]
else:
    class_labels = DEFAULT_CLASS_LABELS

app = FastAPI(title="Lung Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
async def startup_event() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "class_count": len(class_labels),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        content = await file.read()
        pil_img = Image.open(BytesIO(content)).convert("RGB")
        pil_img = pil_img.resize(IMAGE_SIZE)

        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = model.predict(img_arr)
        pred_index = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        if pred_index >= len(class_labels):
            predicted_label = str(pred_index)
        else:
            predicted_label = class_labels[pred_index]

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
            "probabilities": {
                class_labels[i] if i < len(class_labels) else str(i): float(preds[0][i])
                for i in range(preds.shape[1])
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
