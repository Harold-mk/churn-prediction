"""API de prédiction du churn client (Telco).

Charge le pipeline XGBoost réglé + le seuil métier sérialisés en §8 du notebook
(`models/churn_pipeline.joblib`) et expose un endpoint `POST /predict`.

Le pipeline embarque le préprocesseur (imputation + encodage ordinal) : l'API
transmet les features *brutes* du client, sans transformation préalable.
Doc interactive auto-générée : http://localhost:8000/docs
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Chargement du modèle au démarrage (une seule fois) -----------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.joblib"
_artifact = joblib.load(MODEL_PATH)
PIPELINE = _artifact["pipeline"]
THRESHOLD = float(_artifact["threshold"])
FEATURE_ORDER = list(_artifact["feature_order"])


# --- Schémas d'entrée / sortie (validés par Pydantic) -------------------------
class Customer(BaseModel):
    """Features brutes d'un client, telles que dans le dataset Telco."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1, description="1 si senior, 0 sinon")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Ancienneté en mois")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 84.05,
                "TotalCharges": 186.05,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float = Field(description="Probabilité de churn [0, 1]")
    churn: bool = Field(description="True si probabilité >= seuil métier")
    threshold: float = Field(description="Seuil métier appliqué (recall >= 0.80, §6.1)")


# --- Application --------------------------------------------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0",
    description=(
        "Prédit la probabilité de churn d'un client télécom à partir de ses "
        "caractéristiques brutes. Modèle : XGBoost réglé (Optuna), seuil de "
        "décision calibré pour un recall >= 0.80 sur la classe churn."
    ),
)


@app.get("/", tags=["santé"])
def health():
    """Vérification de vie + seuil actif."""
    return {"status": "ok", "model": "xgboost", "threshold": round(THRESHOLD, 4)}


@app.post("/predict", response_model=Prediction, tags=["prédiction"])
def predict(customer: Customer) -> Prediction:
    """Probabilité de churn + classe prédite (au seuil métier) pour un client."""
    # DataFrame à une ligne, colonnes réordonnées comme à l'entraînement.
    df = pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]
    proba = float(PIPELINE.predict_proba(df)[0, 1])
    return Prediction(
        churn_probability=round(proba, 4),
        churn=proba >= THRESHOLD,
        threshold=round(THRESHOLD, 4),
    )
