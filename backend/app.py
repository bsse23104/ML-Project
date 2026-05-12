from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import json

app = FastAPI()

# -------------------
# INPUT SCHEMA (NEW - CLEAN API)
# -------------------
class InputData(BaseModel):
    data: List[float]

# -------------------
# LOAD ARTIFACTS
# -------------------
model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

feature_names = joblib.load("models/feature_names.pkl")

with open("models/optimal_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# -------------------
# HOME ROUTE
# -------------------
@app.get("/")
def home():
    return {"status": "ML API running"}

# -------------------
# PREDICTION ROUTE (CLEAN VERSION)
# -------------------
@app.post("/predict")
def predict(input: InputData):

    try:
        x = input.data
        expected = len(feature_names)

        # -------------------
        # AUTO-PADDING LOGIC
        # -------------------

        # If too short → pad with zeros
        if len(x) < expected:
            x = x + [0] * (expected - len(x))

        # If too long → truncate
        elif len(x) > expected:
            x = x[:expected]

        # Convert to numpy
        X = np.array(x).reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = 1 if prob >= threshold else 0

        return {
            "probability": float(prob),
            "prediction": int(prediction),
            "status": "Failure" if prediction == 1 else "Normal"
        }

    except Exception as e:
        return {"error": str(e)}