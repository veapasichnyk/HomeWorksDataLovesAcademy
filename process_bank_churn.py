from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_features_targets(df: pd.DataFrame, target_col: str = "Exited") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Відділяє ознаки та цільову змінну.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def remove_unwanted_columns(X: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Видаляє неінформативні або ідентифікаційні колонки.
    """
    return X.drop(columns=columns_to_remove, errors='ignore')

def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Повертає списки числових та категоріальних колонок.
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scaler_numeric: bool) -> ColumnTransformer:
    """
    Створює об'єкт препроцесора для числових і категоріальних даних.
    """
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Здійснює повну попередню обробку сирих даних.
    
    :param raw_df: Необроблений датафрейм
    :param scaler_numeric: Чи застосовувати масштабування до числових ознак
    :return: X_train, y_train, X_val, y_val, input_cols, scaler, encoder
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

    return X_train_processed, y_train, X_val_processed, y_val, input_cols, scaler, encoder

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> np.ndarray:
    """
    Обробляє нові дані перед передбаченням, використовуючи збережений scaler та encoder.

    :param new_df: Нові необроблені дані (наприклад, з test.csv)
    :param input_cols: Ознаки, що використовувались під час тренування
    :param scaler: Масштабувальник (може бути None)
    :param encoder: OneHotEncoder для категоріальних ознак
    :return: Масив оброблених даних
    """
    df = new_df.copy()
    df = remove_unwanted_columns(df, ['id', 'CustomerId', 'Surname'])
    df = df[input_cols]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Імпутація
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Масштабування
    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Кодування
    encoded_cats = encoder.transform(df[categorical_cols])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df = df.drop(columns=categorical_cols)
    df_final = pd.concat([df, encoded_cat_df], axis=1)

    return df_final.values
