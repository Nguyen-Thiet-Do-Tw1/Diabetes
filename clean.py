import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Load the diabetes dataset from the provided CSV file
data = pd.read_csv("data/Healthcare-Diabetes.csv")

# Display basic information about the dataset, including data types and non-null counts
print(data.info())

# Generate descriptive statistics for each column in the dataset
print(data.describe())

# Check for duplicate rows in the dataset
duplicate_count = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicated rows from the dataset
data_cleaned = data.drop_duplicates()

# Check for missing (NaN) values in each column
missing_values_count = data.isnull().sum()
print(f"\nCác giá trị rỗng:\n{missing_values_count}")

# Check for NaN or undefined values again after dropping zero-value records
undefined_values_count = data.isna().sum()
print(f"\ncác giá trị không xác định:\n{undefined_values_count}")

# Check for zero values in selected columns
zero_glucose_count = (data['Glucose'] == 0).sum()
zero_bp_count = (data['BloodPressure'] == 0).sum()
zero_skin_count = (data['SkinThickness'] == 0).sum()
zero_insulin_count = (data['Insulin'] == 0).sum()
zero_bmi_count = (data['BMI'] == 0).sum()

# Drop records with zero values in the selected columns
data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['BloodPressure'] == 0].index)
data = data.drop(data[data['SkinThickness'] == 0].index)
data = data.drop(data[data['Insulin'] == 0].index)
data = data.drop(data[data['BMI'] == 0].index)


# Function to plot boxplots for each column to visualize outliers
def plot_boxplots(df, columns):
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot for {column}')
        plt.xlabel(column)
        plt.show()

# List of numerical columns
numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Plot boxplots for numerical columns
plot_boxplots(data, numerical_columns)

# Drop the 'Id' column if it exists
if 'Id' in data.columns:
    data = data.drop(['Id'], axis='columns')

# Reset index
data = data.reset_index(drop=True)

# Output final data and checks
print(f"Cleaned data:\n{data}")
print(f"\nTotal duplicate records removed: {duplicate_count}")
print(f"Number of records with Glucose = 0: {zero_glucose_count}")
print(f"Number of records with BloodPressure = 0: {zero_bp_count}")
print(f"Number of records with SkinThickness = 0: {zero_skin_count}")
print(f"Number of records with Insulin = 0: {zero_insulin_count}")
print(f"Number of records with BMI = 0: {zero_bmi_count}")

from sklearn.preprocessing import StandardScaler

# Xác định các cột cần chuẩn hóa (ngoại trừ cột 'Outcome' là biến mục tiêu)
columns_to_scale = data.columns.difference(['Outcome'])

# Khởi tạo StandardScaler
scaler = StandardScaler()

# Chuẩn hóa các cột được chọn, trừ cột 'Outcome'
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Hiển thị dữ liệu sau khi chuẩn hóa
print(f"\nDữ liệu sau khi chuẩn hóa (trừ cột Outcome):\n{data.head()}")
