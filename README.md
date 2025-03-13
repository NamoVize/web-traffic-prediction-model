# Web Traffic Prediction Model

A user-friendly machine learning application that predicts website traffic trends.

## Features

- 📈 Predict daily and weekly website traffic trends
- 🔮 Visualize future traffic projections
- 📊 Interactive dashboard for data exploration
- 🤖 Pre-trained ML model for immediate predictions
- 📱 Responsive web interface

## Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/NamoVize/web-traffic-prediction-model.git
cd web-traffic-prediction-model

# Run with Docker
docker-compose up
```

Then open your browser to http://localhost:8000

### Option 2: Run with Python

```bash
# Clone the repository
git clone https://github.com/NamoVize/web-traffic-prediction-model.git
cd web-traffic-prediction-model

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser to http://localhost:8000

## How It Works

1. The application loads a pre-trained machine learning model
2. Historical website traffic data is included in the `/data` directory
3. You can view predictions based on this data or upload your own data
4. The model automatically retrains when new data is provided
5. Results are displayed through an interactive dashboard


## Technologies Used

- Python 3.8+
- Scikit-learn for ML modeling
- Prophet for time series forecasting
- Flask for the web server
- Plotly for interactive visualizations
- Docker for containerization

## Data Format

If you want to use your own data, prepare a CSV file with these columns:
- `date`: in YYYY-MM-DD format
- `pageviews`: number of page views for that date

A sample file is provided in `/data/sample_traffic.csv`.

## Project Structure

```
web-traffic-prediction-model/
├── app.py                  # Main Flask application
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Docker build instructions
├── requirements.txt        # Python dependencies
├── data/                   # Sample and training data
├── model/                  # Pre-trained ML models
├── src/                    # Source code
│   ├── predict.py          # Prediction logic
│   ├── train.py            # Model training logic
│   └── utils.py            # Utility functions
├── static/                 # CSS, JS, and images
└── templates/              # HTML templates
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
