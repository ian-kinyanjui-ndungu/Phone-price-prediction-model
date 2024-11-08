from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

def train_model(X_train, y_train):
    """Train the Xiaomi price prediction model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2
    }

def make_prediction(model, scaler, new_data):
    """Make price predictions for new phone data."""
    # Scale the features
    X_new_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)
    return prediction[0]