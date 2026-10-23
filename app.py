import pandas as pd
from fastapi import FastAPI, Depends

from service import PredictionService, get_prediction_service
from schemas import ResponseSchema, TransactionRequestSchema

app = FastAPI()


@app.post("/predict", response_model=ResponseSchema)
def predict(
        payload: TransactionRequestSchema,
        service: PredictionService = Depends(get_prediction_service)
):
    transactions = pd.DataFrame([payload.model_dump()])
    result = service.predict(transactions)

    return result.iloc[0].to_dict()
