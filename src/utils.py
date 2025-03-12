"""
Utility functions for Web Traffic Prediction Model
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'model', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_model(model, filename='model/model.pkl'):
    """Save Prophet model to disk"""
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filename)

def load_model(filename='model/model.pkl'):
    """Load Prophet model from disk"""
    if not os.path.exists(filename):
        # If model doesn't exist, create sample data and train model
        from src.train import train_model
        train_model(get_sample_data())
    
    if os.path.exists(filename):
        try:
            # Load the model
            return joblib.load(filename)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    return None

def get_sample_data():
    """
    Get sample website traffic data
    If sample data doesn't exist, create it
    
    Returns:
        pd.DataFrame: Sample data with 'ds' and 'y' columns
    """
    sample_data_path = 'data/sample_traffic.csv'
    
    # If sample data exists, load it
    if os.path.exists(sample_data_path):
        return pd.read_csv(sample_data_path)
    
    # Otherwise, create synthetic data
    # Generate dates for the past year
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'ds': dates})
    
    # Base traffic (random starting point between 500-1500)
    base = np.random.randint(500, 1500)
    
    # Add trend (gradual increase over time)
    trend = np.linspace(0, 500, len(dates))
    
    # Weekly seasonality (weekends have more traffic)
    weekday = df['ds'].dt.dayofweek
    weekend_boost = (weekday >= 5).astype(int) * 300
    
    # Monthly seasonality
    month_effect = 200 * np.sin(2 * np.pi * df['ds'].dt.day / 30)
    
    # Special events/holidays (random spikes)
    special_events = np.zeros(len(dates))
    event_indices = np.random.choice(len(dates), size=10, replace=False)
    special_events[event_indices] = np.random.randint(300, 800, size=len(event_indices))
    
    # Random noise
    noise = np.random.normal(0, 50, size=len(dates))
    
    # Combine all components
    df['y'] = (base + trend + weekend_boost + month_effect + special_events + noise).astype(int)
    
    # Ensure no negative values
    df['y'] = df['y'].clip(lower=0)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(sample_data_path, index=False)
    
    return df

def format_date(date_str):
    """Format date string to YYYY-MM-DD"""
    return pd.to_datetime(date_str).strftime('%Y-%m-%d')

def calculate_metrics(actual, predicted):
    """
    Calculate error metrics for model evaluation
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        
    Returns:
        dict: Dictionary containing error metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }