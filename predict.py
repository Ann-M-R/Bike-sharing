import pickle
from flask import Flask, request, jsonify

# Load the saved model and DictVectorizer
model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('bike_rentals')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input from the request
    data = request.get_json()
    
    # Transform the input data using the DictVectorizer
    X = dv.transform([data])
    
    # Predict bike rentals (continuous value)
    y_pred = model.predict(X)
    
    result = {
        'predicted_rentals': float(y_pred[0])  # Convert prediction to a float for JSON serialization
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
