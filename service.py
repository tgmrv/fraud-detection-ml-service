from functools import lru_cache

import joblib
import pandas as pd

class PredictionService:
    def __init__(self, model_path: str):
        saved = joblib.load(model_path)

        self.model = saved["model"]
        self.threshold = saved["threshold"]

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        score = self.model.predict_proba(transactions)[:, 1]

        return pd.DataFrame({
            "risk_score": score,
            "is_suspicious": score >= self.threshold,
        })


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService("artifacts/random_forest_model.joblib")