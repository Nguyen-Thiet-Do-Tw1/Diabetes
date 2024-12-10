import os
import io
from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Nạp model và scaler từ file pkl
with open("model/model_randomForest.pkl", 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    scaler = data['scaler']

# Khởi tạo Flask app
app = Flask(__name__)

# Route cho form input
@app.route('/')
def index():
    return render_template('index_file.html')

# Route cho dự đoán và đánh giá hiệu suất
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra nếu file đã được tải lên
    if 'file' not in request.files:
        return render_template('index_file.html', prediction_text="No file uploaded")
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index_file.html', prediction_text="No file selected")

    # Lưu file vào thư mục hiện tại tạm thời
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Đọc file CSV vào DataFrame
    input_data = pd.read_csv(file_path)

    # Sau khi đã đọc xong, xóa file tạm
    os.remove(file_path)

    # Kiểm tra nếu file chứa cột 'Outcome'
    if 'Outcome' in input_data.columns:
        true_labels = input_data['Outcome']  # Lưu nhãn Outcome ban đầu
        input_data_features = input_data.drop(columns=['Outcome'])  # Bỏ cột 'Outcome' cho việc dự đoán
    else:
        return render_template('index_file.html', prediction_text="File test thiếu cột 'Outcome' cho đánh giá hiệu suất.")

    # Chuẩn hóa dữ liệu
    input_data_scaled = scaler.transform(input_data_features)

    # Thực hiện dự đoán
    predictions = model.predict(input_data_scaled)

    # Tính toán hiệu suất
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)

    # Thêm cột dự đoán vào DataFrame ban đầu
    input_data['Predicted_Outcome'] = predictions

    # Tạo file CSV trong bộ nhớ
    csv_output = io.BytesIO()
    input_data.to_csv(csv_output, index=False)
    csv_output.seek(0)  # Đưa con trỏ về đầu bộ nhớ để gửi đi

    # Tô màu các dòng sai
    def highlight_incorrect(row):
        if row['Outcome'] != row['Predicted_Outcome']:
            return ['background-color: #f28b82'] * len(row)  # Màu đỏ nhạt cho toàn bộ dòng
        else:
            return [''] * len(row)

    # Chuyển DataFrame sang HTML với định dạng để tô màu các dòng sai
    table_html = input_data.style.apply(highlight_incorrect, axis=1).to_html()

    # Hiển thị kết quả và bảng
    return render_template(
        'index_file.html',
        prediction_text="Predictions completed!",
        accuracy_text=f'Accuracy: {accuracy:.2f}',
        precision_text=f'Precision: {precision:.2f}',
        recall_text=f'Recall: {recall:.2f}',
        f1_text=f'F1 Score: {f1:.2f}',
        confusion_matrix_text=f'Confusion Matrix: \n{confusion_mat}',
        table_html=table_html,
        download_link='/download_result'  # Đường dẫn để tải file
    ), {'csv_output': csv_output}

# Route để tải file CSV kết quả từ bộ nhớ
@app.route('/download_result')
def download_result():
    # Lấy dữ liệu CSV từ bộ nhớ
    csv_output = request.view_args.get('csv_output')
    
    if csv_output:
        # Gửi file CSV từ bộ nhớ
        return send_file(csv_output, as_attachment=True, download_name="prediction_results.csv", mimetype='text/csv')
    else:
        return "No file available", 400

# Chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)
