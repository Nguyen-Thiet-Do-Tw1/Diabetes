from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Tạo Flask app
app = Flask(__name__)

# Load mô hình đã được huấn luyện và scaler
with open('D:/python/model_randomForest.pkl', 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    scaler = model_data['scaler']

# Route cho form input
@app.route('/')
def index():
    return render_template('index_form.html')

# Route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    data = request.form.to_dict()
    features = np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]], dtype=float)

    # Chuẩn hóa dữ liệu
    features_scaled = scaler.transform(features)

    # Dự đoán kết quả và xác suất
    prediction = model.predict(features_scaled)[0]
    # Dự đoán dạng nhị phân (0 hoặc 1)
    outcome = "Positive" if prediction == 1 else "Negative"
    message = "You may have diabetes" if prediction == 1 else "You may not have diabetes"

    return render_template('index_form.html', prediction_text=message, data=data)

if __name__ == '__main__':
    app.run(debug=True)
