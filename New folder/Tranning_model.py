import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Đọc dữ liệu
df = pd.read_csv("data/train_dataset.csv", sep=',')

# Chia dữ liệu thành X và y
x = df.drop('Outcome', axis='columns')  # Loại bỏ cột Outcome
y = df['Outcome']  # Lấy cột Outcome làm nhãn

# Chuẩn hóa dữ liệu (nếu cần)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Chia tập huấn luyện và kiểm tra
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest với các tham số đã điều chỉnh
model = RandomForestClassifier(
    n_estimators=100,        # Số cây trong rừng
    max_depth=10,            # Độ sâu tối đa của mỗi cây
    min_samples_split=10,    # Số lượng mẫu tối thiểu để chia tách một nút
    min_samples_leaf=5,      # Số lượng mẫu tối thiểu ở mỗi lá
    random_state=42          # Đảm bảo tính tái lập
)
model.fit(xtrain, ytrain)

# Đánh giá mô hình trên tập huấn luyện và tập kiểm tra
train_accuracy = model.score(xtrain, ytrain)
test_accuracy = model.score(xtest, ytest)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Đánh giá mô hình với confusion matrix
ypred = model.predict(xtest)
conf_matrix = confusion_matrix(ytest, ypred)
print("Confusion Matrix:\n", conf_matrix)

# Tính thêm các chỉ số precision, recall, F1-score
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

import pickle

# Tạo một dictionary để lưu cả model và scaler
model_scaler = {
    'model': model,
    'scaler': scaler
}

# Lưu vào file .pkl
with open("model\model_randomForest_test.pkl", 'wb') as file:
    pickle.dump(model_scaler, file)

print("Model và scaler đã được lưu thành công!")
