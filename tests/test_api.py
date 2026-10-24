from tests.conftest import TRANSACTION


def test_predict_response(client):
    response = client.post("/predict", json=TRANSACTION)

    assert response.status_code == 200
    assert response.json() == {
            "risk_score": 0.25,
            "is_suspicious": True,
        }

def test_predict_requires_amount(client):
    transaction = TRANSACTION.copy()
    transaction.pop("Amount")

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422

    errors = response.json()["detail"]
    assert errors[0]["loc"] == ["body", "Amount"]
    assert errors[0]["type"] == "missing"