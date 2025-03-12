"""
Training module for Web Traffic Prediction Model
"""
import os
import pandas as pd
from prophet import Prophet
from src.utils import save_model, get_sample_data

def train_model(data=None, periods=30):
    """
    Train a Prophet model for web traffic prediction
    
    Args:
        data (pd.DataFrame, optional): Training data with 'ds' and 'y' columns
        periods (int): Number of days to include in future predictions
        
    Returns:
        Prophet: Trained Prophet model
    """
    # If no data provided, use sample data
    if data is None:
        data = get_sample_data()
    
    # Ensure data has correct columns
    if 'ds' not in data.columns or 'y' not in data.columns:
        raise ValueError("Data must contain 'ds' and 'y' columns")
    
    # Convert dates to datetime if they're not already
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Initialize and train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add additional regressors if needed
    # Add day of week as a regressor
    data['day_of_week'] = data['ds'].dt.dayofweek
    model.add_regressor('day_of_week')
    
    # Fit the model
    model.fit(data)
    
    # Save the model
    save_model(model)
    
    return model

def evaluate_model(data=None):
    """
    Evaluate the model performance using cross-validation
    
    Args:
        data (pd.DataFrame, optional): Data with 'ds' and 'y' columns
        
    Returns:
        pd.DataFrame: Cross-validation results
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    
    # If no data provided, use sample data
    if data is None:
        data = get_sample_data()
    
    # Train model
    model = train_model(data)
    
    # Perform cross-validation
    df_cv = cross_validation(
        model, 
        initial='30 days', 
        period='7 days',
        horizon='14 days'
    )
    
    # Calculate performance metrics
    df_p = performance_metrics(df_cv)
    
    return df_p