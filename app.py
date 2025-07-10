print("🚨 CHECKPOINT: app.py started running 🚨")
print("👋 app.py is being executed...")

from flask import Flask, request, jsonify
import joblib
import numpy as np

print("✅ Flask and modules imported")

app = Flask(__name__)

print("📦 Loading the model...")
model = joblib.load('fraud_model.pkl')  # ✅ correct path inside app/


print("✅ Model loaded successfully")

@app.route('/')
def home():
    return "🚀 Credit Card Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['data']).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({'fraud': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🔥 Starting Flask server...")
    app.run(debug=True)
