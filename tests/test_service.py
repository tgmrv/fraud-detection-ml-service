import pandas as pd

from service import PredictionService
from tests.conftest import FakeModel


def test_predict_service(monkeypatch):
    saved = {
        "model": FakeModel(),
        "threshold": 0.1,
    }

    monkeypatch.setattr("service.joblib.load", lambda path: saved)

    service = PredictionService("test_model.joblib")

    transactions = pd.DataFrame({"Amount": [10.0, 20.0, 30.0]})

    result = service.predict(transactions)

    assert result["risk_score"].tolist() == [0.05, 0.1, 0.3]
    assert result["is_suspicious"].tolist() == [False, True, True]
