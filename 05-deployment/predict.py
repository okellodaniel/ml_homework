import pickle
import logging
from flask import request, jsonify, Flask

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log receipt of request
        logging.info("Received prediction request.")
        
        # Get JSON data from request
        customer = request.get_json()
        logging.info(f"Request data: {customer}")
        
        # Generate prediction
        prediction = model_prediction(customer)
        churn = prediction >= 0.5
        
        # Log prediction result
        logging.info(f"Prediction: {prediction}, Churn: {churn}")

        return jsonify({
            'prediction': float(prediction),
            'churn': bool(churn)
        })
    except Exception as e:
        logging.error("Error occurred during prediction", exc_info=True)
        return jsonify({'error': str(e)}), 500


def model_prediction(customer):
    model, dv = model_dv()
    X = dv.transform([customer])
    prediction = model.predict_proba(X)[:,1]
    print('prediction',prediction)
    return prediction


def model_dv():
    try:
        # Log loading of model and transformer
        logging.info("Loading model and transformer.")
        
        with open('model1.bin', 'rb') as model_in:
            model = pickle.load(model_in)
        
        with open('dv.bin', 'rb') as dv_file:
            dv = pickle.load(dv_file)
        
        logging.info("Model and transformer loaded successfully.")
        return model, dv
    except FileNotFoundError as e:
        logging.error("File not found. Ensure model1.bin and dv.bin are present.", exc_info=True)
        raise
    except Exception as e:
        logging.error("Error loading model and transformer", exc_info=True)
        raise


if __name__ == '__main__':
    # Log server start
    logging.info("Starting Flask app.")
    app.run(host='0.0.0.0', port=9696)
