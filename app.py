#!/usr/bin/env python
"""
Main Flask application for Web Traffic Prediction Model
"""
import os
import io
import base64
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from src.predict import predict_traffic
from src.train import train_model
from src.utils import load_model, save_model, get_sample_data, create_directories

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-flask-app')

# Create necessary directories on startup
create_directories()

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard with current data and predictions"""
    # Load sample data
    df = get_sample_data()
    
    # Ensure we have a trained model
    model_exists = os.path.exists('model/model.pkl')
    if not model_exists:
        train_model(df, periods=30)
    
    # Get predictions for the next 30 days
    future = predict_traffic(days=30)
    
    # Prepare data for visualization
    df['ds'] = pd.to_datetime(df['ds'])
    future['ds'] = pd.to_datetime(future['ds'])
    
    # Create Plotly figure for historical data
    fig1 = px.line(df, x='ds', y='y', title='Historical Website Traffic')
    fig1.update_layout(xaxis_title='Date', yaxis_title='Page Views')
    graph1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create Plotly figure for future predictions
    fig2 = px.line(future, x='ds', y='yhat', title='Predicted Website Traffic')
    fig2.add_scatter(x=future['ds'], y=future['yhat_lower'], fill=None, mode='lines', 
                     line_color='rgba(0,100,80,0.2)', name='Lower Bound')
    fig2.add_scatter(x=future['ds'], y=future['yhat_upper'], fill='tonexty', mode='lines',
                     line_color='rgba(0,100,80,0.2)', name='Upper Bound')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Predicted Page Views')
    graph2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create a combined view
    combined = pd.concat([
        df[['ds', 'y']].rename(columns={'y': 'Historical'}),
        future[['ds', 'yhat']].rename(columns={'yhat': 'Predicted'})
    ], axis=0)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name='Predicted', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=future['ds'], y=future['yhat_upper'], 
                             fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper Bound'))
    fig3.add_trace(go.Scatter(x=future['ds'], y=future['yhat_lower'], 
                             fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower Bound'))
    fig3.update_layout(title='Historical and Predicted Website Traffic',
                      xaxis_title='Date', yaxis_title='Page Views',
                      hovermode='x unified')
    graph3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Get some quick stats
    last_value = df['y'].iloc[-1]
    avg_value = df['y'].mean()
    max_pred = future['yhat'].max()
    min_pred = future['yhat'].min()
    trend = 'Increasing' if future['yhat'].iloc[-1] > future['yhat'].iloc[0] else 'Decreasing'
    
    stats = {
        'last_value': int(last_value),
        'avg_value': int(avg_value),
        'max_pred': int(max_pred),
        'min_pred': int(min_pred),
        'trend': trend
    }
    
    return render_template('dashboard.html', 
                          graph1=graph1_json, 
                          graph2=graph2_json,
                          graph3=graph3_json,
                          stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests"""
    if request.method == 'POST':
        days = int(request.form.get('days', 30))
        future = predict_traffic(days=days)
        
        # Create visualization
        fig = px.line(future, x='ds', y='yhat', title=f'Traffic Prediction for Next {days} Days')
        fig.add_scatter(x=future['ds'], y=future['yhat_lower'], fill=None, mode='lines', 
                        line_color='rgba(0,100,80,0.2)', name='Lower Bound')
        fig.add_scatter(x=future['ds'], y=future['yhat_upper'], fill='tonexty', mode='lines',
                        line_color='rgba(0,100,80,0.2)', name='Upper Bound')
        fig.update_layout(xaxis_title='Date', yaxis_title='Predicted Page Views')
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('predict.html', graph=graph_json, days=days, 
                              prediction_data=future.to_dict(orient='records'))
    
    return render_template('predict.html', days=30)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle data upload and model retraining"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if user didn't select a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            try:
                # Read CSV file
                df = pd.read_csv(file)
                
                # Check if required columns exist
                if 'date' not in df.columns or 'pageviews' not in df.columns:
                    flash('CSV must contain "date" and "pageviews" columns')
                    return redirect(request.url)
                
                # Prepare data for training
                df = df.rename(columns={'date': 'ds', 'pageviews': 'y'})
                
                # Train model
                train_model(df)
                
                flash('Model successfully trained with your data!')
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.json
    days = data.get('days', 30)
    
    future = predict_traffic(days=days)
    result = future.to_dict(orient='records')
    
    return jsonify({
        'success': True,
        'prediction': result
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)