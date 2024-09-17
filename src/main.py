# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from .model import load_model, evaluate_model, train_model, save_model
# from .data_utils import load_data, preprocess_data
# from .explanations import global_shap_explanation, local_lime_explanation

# app = FastAPI()

# class PredictionInput(BaseModel):
#     file_path: str
#     target_column: str

# class PredictRequest(BaseModel):
#     age: int
#     income: float
#     tenure: int
#     gender: str
#     contract: str

# @app.post("/train/")
# async def train_endpoint(input_data: PredictionInput):
#     try:
#         # Load data
#         df = load_data(input_data.file_path)
#         # Preprocess data
#         X, y = preprocess_data(df, input_data.target_column)
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         # Train model
#         model = train_model(X_train, y_train)
#         save_model(model, "churn_model.pkl")
#         # Evaluate model
#         metrics = evaluate_model(model, X_test, y_test)
#         return {"model_metrics": metrics}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

# @app.post("/predict/")
# async def predict_endpoint(request: PredictRequest):
#     try:
#         # Convert request data to DataFrame
#         data = {
#             'age': [request.age],
#             'income': [request.income],
#             'tenure': [request.tenure],
#             'gender': [request.gender],
#             'contract': [request.contract]
#         }
#         df = pd.DataFrame(data)

#         # Load the model
#         model = load_model("churn_model.pkl")

#         # Preprocess the data
#         X, _ = preprocess_data(df)

#         # Make predictions
#         predictions = model.predict(X)
#         probability = model.predict_proba(X)[:, 1]  # Assuming binary classification

#         return {
#             "prediction": "Churn" if predictions[0] == 1 else "Not Churn",
#             "probability": float(probability[0])
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# @app.post("/explain/")
# async def explain_endpoint(file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)
#         model = load_model("churn_model.pkl")
#         X, _ = preprocess_data(df)
#         shap_values = global_shap_explanation(model, X)
#         return {"shap_values": shap_values}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from .model import load_model, evaluate_model, train_model, save_model
from .data_utils import load_data, preprocess_data
from .explanations import global_shap_explanation, local_lime_explanation

app = FastAPI()

class PredictionInput(BaseModel):
    file_path: str
    target_column: str



@app.post("/train/")
async def train_endpoint(input_data: PredictionInput):
    try:
        # Load data
        df = load_data(input_data.file_path)
        # Preprocess data
        X, y = preprocess_data(df, input_data.target_column)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train model
        model = train_model(X_train, y_train)
        save_model(model, "churn_model.pkl")
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        return {"model_metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

class PredictRequest(BaseModel):
    rownumber: Optional[int]
    age: int
    income: float      # This corresponds to estimatedsalary
    tenure: int
    gender: str
    contract: str      # This corresponds to geography
    customerid: int
    surname: str
    creditscore: int
    balance: float
    numofproducts: int
    hascrcard: int
    isactivemember: int
    rownumber: int     # Add rownumber to the PredictRequest


@app.post("/predict/")
async def predict_endpoint(request: PredictRequest):
    try:
        # Convert request data to DataFrame
        data = {
            'rownumber': [request.rownumber],  # Add rownumber
            'age': [request.age],
            'estimatedsalary': [request.income],
            'creditscore': [request.creditscore],  # Fix this to use the right field
            'gender': [request.gender],
            'geography': [request.contract],
            'balance': [request.balance],
            'customerid': [request.customerid],
            'hascrcard': [request.hascrcard],
            'isactivemember': [request.isactivemember],
            'numofproducts': [request.numofproducts]
            # Add any additional required features here
        }
        df = pd.DataFrame(data)

        # Load the model
        model = load_model("churn_model.pkl")

        # Preprocess the data (no target column for prediction)
        X, _ = preprocess_data(df, target_column=None)

        # Make predictions
        predictions = model.predict(X)
        probability = model.predict_proba(X)[:, 1]  # Assuming binary classification

        return {
            "prediction": "Churn" if predictions[0] == 1 else "Not Churn",
            "probability": float(probability[0])
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {traceback_str}")






@app.post("/explain/")
async def explain_endpoint(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        model = load_model("churn_model.pkl")
        X, _ = preprocess_data(df)
        shap_values = global_shap_explanation(model, X)
        return {"shap_values": shap_values}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")
