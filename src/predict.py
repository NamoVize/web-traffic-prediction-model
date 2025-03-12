"""
Prediction module for Web Traffic Prediction Model
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from src.utils import load_model, get_sample_data

def predict_traffic(days=30, custom_data=None):
    """
    Predict website traffic for the specified number of days
    
    Args:
        days (int): Number of days to predict
        custom_data (pd.DataFrame, optional): Custom data to base prediction on
        
    Returns:
        pd.DataFrame: Dataframe containing prediction results
    """
    # Load the model
    model = load_model()
    
    # If no model exists, return empty dataframe
    if model is None:
        return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
    
    # Create future dataframe
    future_dates = pd.date_range(
        start=datetime.now().date(),
        periods=days,
        freq='D'
    )
    future = pd.DataFrame({'ds': future_dates})
    
    # Make prediction
    forecast = model.predict(future)
    
    # Select relevant columns
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Round prediction values to integers
    result['yhat'] = result['yhat'].round().astype(int)
    result['yhat_lower'] = result['yhat_lower'].round().astype(int)
    result['yhat_upper'] = result['yhat_upper'].round().astype(int)
    
    return result

def predict_with_custom_data(custom_data, days=30):
    """
    Train a model with custom data and make predictions
    
    Args:
        custom_data (pd.DataFrame): Custom data with 'ds' and 'y' columns
        days (int): Number of days to predict
        
    Returns:
        pd.DataFrame: Dataframe containing prediction results
    """
    from src.train import train_model
    
    # Train model with custom data
    train_model(custom_data)
    
    # Make prediction
    return predict_traffic(days=days)