from flask import Flask, request, jsonify
import pickle

app = Flask(__name)

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Get input features from the request
        data = request.get_json()
        features = [
            data['bedrooms'],
            data['bathrooms'],
            data['sqft_living'],
            data['sqft_lot'],
            data['floors'],
            data['waterfront'],
            data['view'],
            data['condition'],
            data['sqft_above'],
            data['sqft_basement']
        ]

        # Make a prediction using the loaded model
        predicted_price = model.predict([features])

        return jsonify({'predicted_price': predicted_price[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
