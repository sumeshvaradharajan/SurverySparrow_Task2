# Generic Churn Prediction

## Overview
This project implements a churn prediction model using machine learning techniques. The system allows users to train a model on customer data and make predictions on customer churn based on various features.

The churn prediction application integrates several key techniques and components to function effectively. The FastAPI framework (main.py) is used to create a RESTful API with endpoints for model training, prediction, and explanation. Data preprocessing, including handling missing values and encoding categorical variables, is performed using functions from data_utils.py, which employs techniques like label encoding and standard scaling. For model training and evaluation, model.py utilizes Scikit-learn methods such as train-test splitting and random forests, with model metrics like accuracy, precision, recall, and ROC-AUC computed for performance assessment. The model is saved and loaded using pickle. The /predict/ endpoint processes input data, ensuring it matches the trained model's feature requirements. For generating explanations, explanations.py employs SHAP (SHapley Additive exPlanations) to provide global insights into feature importances, which helps in understanding the model's decisions. This integrated approach ensures robust data handling, accurate predictions, and transparent model explanations.

# Technologies Used
FastAPI: Framework for building APIs quickly.
Pandas: Data manipulation and analysis.
Scikit-learn: Machine learning algorithms and model evaluation.
SHAP: Explainable AI for feature importance.

 
