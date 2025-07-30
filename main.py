#main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from utils.sentinel import get_satellite_mean_nutrient, get_satellite_features
from models.predictor import predict_nutrients

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeatureRequest(BaseModel):
    latitude: float
    longitude: float
    date: str

@app.get("/")
def root():
    return {"message": "Soil Nutrient Predictor is running!"}

@app.post("/extract_features")
def extract_features(req: FeatureRequest):
    try:
        features = get_satellite_features(req.latitude, req.longitude, req.date)
        print("Returning features:", features)  # Add this for debugging
        return features
    except Exception as e:
        return {"error": str(e)}

class PredictionRequest(BaseModel):
    nutrients: List[str]
    data: Dict[str, Any]

@app.post("/predict_nutrients")
def predict(req: PredictionRequest):
    print("Running Prediction")
    return predict_nutrients(req.dict())

class TimeSeriesPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    date: str
    nutrients: List[str]

@app.post("/predict_nutrient_timeseries")
def predict_from_timeseries(req: TimeSeriesPredictionRequest):
    try:
        result = get_satellite_mean_nutrient(req.latitude, req.longitude, req.date, req.nutrients)
        return result
    except Exception as e:
        return {"error": str(e)}
