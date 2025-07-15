from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

def split_features_targets(df: pd.DataFrame, target_col: str = "Exited") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features and target variable.

    :param df: Input DataFrame.
    :param target_col: Name of the target column.
    :return: Tuple of features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def remove_unwanted_columns(X: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame.

    :param X: Input feature DataFrame.
    :param columns_to_remove: List of columns to remove.
    :return: DataFrame without the unwanted columns.
    """
    return X.drop(columns=columns_to_remove, errors='ignore')

def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns in the DataFrame.

    :param X: Input feature DataFrame.
    :return: Tuple containing list of numeric and categorical column names.
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scaler_numeric: bool) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline for numeric and categorical features.

    :param numeric_cols: List of numeric column names.
    :param categorical_cols: List of categorical column names.
    :param scaler_numeric: Whether to apply StandardScaler to numeric features.
    :return: ColumnTransformer object.
    """
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder, List[str]]:
    """
    Performs full preprocessing of raw data.

    :param raw_df: Raw input DataFrame.
    :param scaler_numeric: Whether to apply scaling to numeric features.
    :return:
        - X_train_processed: Preprocessed training features.
        - y_train: Training labels.
        - X_val_processed: Preprocessed validation features.
        - y_val: Validation labels.
        - input_cols: Original feature names.
        - scaler: Fitted scaler or None.
        - encoder: Fitted encoder.
        - final_feature_names: Transformed feature names.
    """
    columns_to_remove = ['id', 'CustomerId', 'Surname']
    target_col = 'Exited'

    X, y = split_features_targets(raw_df, target_col)
    X = remove_unwanted_columns(X, columns_to_remove)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numeric_cols, categorical_cols = get_column_types(X_train)
    input_cols = numeric_cols + categorical_cols

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_numeric)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    scaler = preprocessor.named_transformers_["num"].named_steps.get("scaler") if scaler_numeric else None
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
    final_feature_names = numeric_cols + cat_features

    return X_train_processed, y_train, X_val_processed, y_val, input_cols, scaler, encoder, final_feature_names

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> np.ndarray:
    """
    Preprocesses new/unseen data for prediction using fitted scaler and encoder.

    :param new_df: New data in the form of a DataFrame.
    :param input_cols: List of original feature columns used in training.
    :param scaler: Fitted StandardScaler object.
    :param encoder: Fitted OneHotEncoder object.
    :return: Numpy array of processed features ready for prediction.
    """
    df = new_df.copy()

    df = remove_unwanted_columns(df, ['id', 'CustomerId', 'Surname'])
    df = df[input_cols]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    encoded_cats = encoder.transform(df[categorical_cols])
    if issparse(encoded_cats):
        encoded_cats = encoded_cats.toarray()

    encoded_cat_df = pd.DataFrame(encoded_cats, index=df.index)

    df = df.drop(columns=categorical_cols)
    df_final = pd.concat([df.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

    return df_final.values
