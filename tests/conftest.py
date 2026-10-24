import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from app import app
from service import get_prediction_service

TRANSACTION = {
    "Time": 0.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 10.0,
}


class FakePredictionService:
    def predict(self, transactions):
        return pd.DataFrame({
            "risk_score": [0.25],
            "is_suspicious": [True],
        })


class FakeModel:
    def predict_proba(self, transactions):
        return np.array([
            [0.95, 0.05],
            [0.90, 0.10],
            [0.70, 0.30],
        ])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(
        app.dependency_overrides,
        get_prediction_service,
        lambda: FakePredictionService(),
    )

    with TestClient(app) as test_client:
        yield test_client