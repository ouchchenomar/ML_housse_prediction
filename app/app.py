"""
Flask API for housing price prediction.
Provides endpoints for predicting housing prices and checking service health.
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and feature names
model = None
feature_names = []

def load_model():
    global model, feature_names
    # Use absolute paths instead of relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(current_dir), 'model')
    model_path = os.path.join(model_dir, 'model.pkl')
    
    print(f"Looking for model at: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        # If not found, try alternate paths
        alt_model_path = os.environ.get('MODEL_PATH', 'model.pkl')
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
            print(f"Found model at alternate path: {model_path}")
        else:
            print(f"Model not found at {model_path} or {alt_model_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            raise FileNotFoundError(f"Model file not found. Please ensure model.pkl exists.")
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        # For the California Housing dataset, use default feature names
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        print(f"Using feature names: {feature_names}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify the service is running."""
    if model is not None:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Model not loaded"}), 500

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict housing price based on input features.
    Expected JSON format:
    {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    """
    global model
    # Load model if not already loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Could not load model: {str(e)}"}), 500
        
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Create feature array in the correct order
        features = []
        missing_features = []
        
        for feature in feature_names:
            if feature in data:
                features.append(data[feature])
            else:
                missing_features.append(feature)
                features.append(0)  # Default value
        
        # Warn about missing features
        if missing_features:
            print(f"Warning: Missing features in input: {missing_features}")
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        
        # Return prediction
        result = {
            "prediction": float(prediction[0]),  # Convert numpy float to Python float for JSON serialization
            "missing_features": missing_features
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# If running directly (not through gunicorn/wsgi)
if __name__ == '__main__':
    # Try to load the model on startup
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {str(e)}")
        print("Will try to load model when prediction endpoint is accessed.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)