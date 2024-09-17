import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the given file path.
    :param file_path: Path to the dataset (csv file).
    :return: Loaded pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    df = df.iloc[:, 1:] 
    return df

def preprocess_data(df: pd.DataFrame, target_column: str = 'churn') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by encoding categorical variables, scaling numerical data, and handling missing values.
    :param df: The DataFrame with raw data.
    :param target_column: The name of the target column.
    :return: Tuple of processed features (X) and target (y).
    """
    # Handle missing values by filling them with median for numeric columns and mode for categorical
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)

    # Separate features and target variable
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Label encoding for categorical variables
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y


