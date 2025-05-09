from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# 🔄 CHANGE THIS: Load your trained model
model = joblib.load(r"C:\Users\RISHIKA\rishika code\Downloads\crop_yield_train\yield_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # 🔄 CHANGE THIS: Use the exact order of features used in training
        input_features = [
            int(data['State']),
            int(data['Crop_Year']),
            int(data['Season']),
            float(data['Area']),
            int(data['Crop']),
            float(data['Production'])
        ]
        prediction = model.predict([input_features])[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
