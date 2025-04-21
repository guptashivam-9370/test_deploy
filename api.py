from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from preprocessing import preprocess_input_data

app = Flask(__name__)

# Global variables to store model and preprocessing objects
MODEL = None
SCALER = None
FEATURE_COLUMNS = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global MODEL, SCALER, FEATURE_COLUMNS
    
    # Load model
    model_path = os.environ.get('MODEL_PATH', 'saved_models/catboost_regressor.joblib')
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load scaler
    scaler_path = os.environ.get('SCALER_PATH', 'saved_models/scaler.joblib')
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Load feature columns
    feature_columns_path = os.environ.get('FEATURE_COLUMNS_PATH', 'saved_models/feature_columns.joblib')
    if os.path.exists(feature_columns_path):
        FEATURE_COLUMNS = joblib.load(feature_columns_path)
    else:
        raise FileNotFoundError(f"Feature columns file not found: {feature_columns_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        return jsonify({
            'status': 'error',
            'message': 'Model, scaler, or feature columns not loaded'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'message': 'API is running and model is loaded'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    
    Request body should contain the input features
    """
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load model: {str(e)}'
            }), 500
    
    # Get input data from request
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        # Preprocess input data
        preprocessed_input = preprocess_input_data(input_data, SCALER, FEATURE_COLUMNS)
        
        # Make prediction
        prediction = MODEL.predict(preprocessed_input)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': float(prediction),
            'model_used': type(MODEL).__name__
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Try to load model at startup
    try:
        load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")
        print("Model will be loaded on first prediction request")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)
