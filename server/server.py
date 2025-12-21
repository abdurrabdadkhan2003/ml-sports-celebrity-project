"""
Flask Web Server for Sports Celebrity Image Classification

This module serves the trained ML model through a REST API and provides
a user-friendly web interface for uploading and classifying sports celebrity images.

Author: Abdurrab Dadkhan
Date: December 2024
"""

from flask import Flask, request, jsonify
import util

# Initialize Flask application
app = Flask(__name__)


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    """
    API endpoint for classifying uploaded images of sports celebrities.
    
    Request Methods:
        - GET: Returns API documentation
        - POST: Accepts image file and returns predictions
    
    Request Data (POST):
        - image_data: Base64 encoded image file from HTML form
    
    Returns:
        JSON response containing:
        - 'class': Predicted celebrity name (string)
        - 'class_probability': Confidence score (0-100 scale)
        - 'class_dictionary': Mapping of all class probabilities
    
    Example Response:
        {
            "class": "lionel_messi",
            "class_probability": 85.5,
            "class_dictionary": {
                "lionel_messi": 85.5,
                "roger_federer": 8.2,
                "virat_kohli": 4.1,
                "maria_sharapova": 1.8,
                "serena_williams": 0.4
            }
        }
    """
    
    # Extract image data from request form
    image_data = request.form['image_data']
    
    # Pass to utility function for prediction
    # The util module handles: preprocessing, feature extraction, and model inference
    response = jsonify(util.classify_image(image_data))
    
    # Add CORS headers to allow cross-origin requests from web interface
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response


if __name__ == "__main__":
    # Print startup message
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    
    # Load pre-trained model and artifacts from disk
    # This initializes global variables in util.py for faster predictions
    util.load_saved_artifacts()
    
    # Start Flask development server
    # Change debug=True for production settings
    # Port 5000: Standard development port, accessible at http://127.0.0.1:5000
    app.run(port=5000)
