
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ======================================================
# Heart Disease Prediction by Machine Learning
# ======================================================

app = FastAPI(
    title="Heart Disease Prediction by Machine Learning",
    description="Professional Machine Learning API for Heart Disease Prediction",
    version="1.0.0"
)

# ======================================================
# Load Trained Model
# ======================================================

MODEL_PATH = "heart_disease_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Unable to load model: {e}")


# ======================================================
# Input Schema
# ======================================================

class HeartDiseaseInput(BaseModel):

    age: int
    gender: int
    region: int
    income_level: int

    hypertension: int
    diabetes: int

    cholesterol_level: float
    obesity: int
    waist_circumference: float
    family_history: int

    smoking_status: int
    alcohol_consumption: int
    physical_activity: int
    dietary_habits: int
    air_pollution_exposure: int
    stress_level: int

    sleep_hours: float

    blood_pressure_systolic: float
    blood_pressure_diastolic: float

    fasting_blood_sugar: float
    cholesterol_hdl: float
    cholesterol_ldl: float
    triglycerides: float

    EKG_results: int

    previous_heart_disease: int
    medication_usage: int
    participated_in_free_screening: int
  # ======================================================
# Prediction Endpoint
# ======================================================

@app.post("/predict")
def predict(data: HeartDiseaseInput):

    try:
        # Convert input into model format
        features = np.array([[
            data.age,
            data.gender,
            data.region,
            data.income_level,
            data.hypertension,
            data.diabetes,
            data.cholesterol_level,
            data.obesity,
            data.waist_circumference,
            data.family_history,
            data.smoking_status,
            data.alcohol_consumption,
            data.physical_activity,
            data.dietary_habits,
            data.air_pollution_exposure,
            data.stress_level,
            data.sleep_hours,
            data.blood_pressure_systolic,
            data.blood_pressure_diastolic,
            data.fasting_blood_sugar,
            data.cholesterol_hdl,
            data.cholesterol_ldl,
            data.triglycerides,
            data.EKG_results,
            data.previous_heart_disease,
            data.medication_usage,
            data.participated_in_free_screening
        ]])

        # Prediction
        prediction = int(model.predict(features)[0])

        # Probability (if supported)
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        else:
            probability = None

        # Result
        result = (
            "High Risk of Heart Disease"
            if prediction == 1
            else "Low Risk of Heart Disease"
        )

        return {
            "project": "Heart Disease Prediction by Machine Learning",
            "prediction": prediction,
            "result": result,
            "probability": round(probability, 4) if probability is not None else None,
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ======================================================
# Root Endpoint
# ======================================================

@app.get("/")
def home():
    return {
        "project": "Heart Disease Prediction by Machine Learning",
        "version": "1.0.0",
        "message": "Welcome to the Heart Disease Prediction by Machine Learning API.",
        "documentation": "/docs",
        "health": "/health"
    }


# ======================================================
# Health Check Endpoint
# ======================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ======================================================
# Run the Application
# ======================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
