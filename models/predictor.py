# predictor.py

import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "Saran08/soil_nutrition_prediction"  # <- CHANGE THIS to your HF model repo

# Filenames for models and selectors
NUTRIENTS = ["N", "P", "K", "OC", "pH"]

# Define model and selector filenames
MODEL_PATHS = {
    nutrient: hf_hub_download(REPO_ID, filename=f"rf_model_{nutrient}.pkl")
    for nutrient in NUTRIENTS
}

SELECTOR_PATHS = {
    nutrient: hf_hub_download(REPO_ID, filename=f"rf_selector_{nutrient}.pkl")
    for nutrient in NUTRIENTS
}

# Load all models and selectors into memory
MODELS = {
    nutrient: joblib.load(path)
    for nutrient, path in MODEL_PATHS.items()
}

SELECTORS = {
    nutrient: joblib.load(path)
    for nutrient, path in SELECTOR_PATHS.items()
}

FEATURES = {
    nutrient: ["B8", "B4", "B5", "B11", "B9", "B1", "SR_n2", "SR_N", "TBVI1", "NDWI",
               "NDVI_G", "PSRI", "NDVIRE1n", "NDVIRE2n", "NDVIRE3n", "BI", "CI",
               "SI", "latitude", "longitude", "B8_minus_B4", "NDVI_G_times_PSRI"]
    for nutrient in NUTRIENTS
}


def predict_nutrients(payload):
    nutrients = payload.get("nutrients", [])
    data = payload.get("data", {})
    predictions = {}

    for nutrient in nutrients:
        model = MODELS.get(nutrient)
        selector = SELECTORS.get(nutrient)
        feature_list = FEATURES[nutrient]

        input_vector = [data.get(f, 0) for f in feature_list]
        X = selector.transform([input_vector])  # shape: (1, 4)
        prediction = model.predict(X)[0]
        predictions[nutrient] = round(float(prediction), 3)

    return predictions
