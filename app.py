from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler using your exact file names
with open('model (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler (1).pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract features (Expecting a list of 4 numbers)
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale the incoming features using the loaded scaler
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        prediction = model.predict(scaled_features)
        
        # Return the result as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(debug=True, port=5000)
