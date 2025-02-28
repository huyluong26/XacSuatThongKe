import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc File
file_path = "CNTT17-01_LuongQuangHuy_BaiKTSo2.xls"
df=pd.read_excel(file_path)
# # Đọc File theo kiểu đã tóm tắt dữ liệu
# print(df.describe())
# Đọc File nguyên dạng
print(df)
# 1. làm sạch file
#Chuyển đổi giá trị trong cột 'Điểm thi' thành kiểu số
df['Điểm thi'] = pd.to_numeric(df['Điểm thi'], errors='coerce')
#Tính giá trị trung bình của điểm thi, bỏ qua các giá trị NaN
average = df['Điểm thi'].mean()
#Thay thế các giá trị NaN bằng giá trị trung bình
df.fillna({'Điểm thi':average}, inplace=True)

#Chuyển đổi giá trị trong cột 'Điểm tổng kết' thành kiểu số
df['Điểm tổng kết'] = pd.to_numeric(df['Điểm tổng kết'], errors='coerce')
#Tính giá trị trung bình điểm tổng kết, bỏ qua các giá trị NaN
average = df['Điểm tổng kết'].mean()
#Thay thế các giá trị NaN bằng giá trị trung bình
df.fillna({'Điểm tổng kết':average}, inplace=True)
print(df)

#Vẽ biểu đồ cột 
plt.figure(figsize=(12, 6))  # Kích thước biểu đồ
plt.bar(df['Mã sinh viên'], df['Điểm tổng kết'], color='skyblue')
plt.title('Điểm tổng kết của các sinh viên', fontsize=14)
plt.xlabel('Mã sinh viên', fontsize=12)
plt.ylabel('Điểm tổng kết', fontsize=12)
plt.ylim(0, 10)
plt.tight_layout() 
plt.show()

## Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(df)

# Tách biến độc lập (X) và phụ thuộc (y)
X = df["Điểm thi"].values.reshape(-2, 1)
y = df["Điểm tổng kết"].values

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# Dự đoán giá trị
y_pred = model.predict(X)

# Tính toán các hệ số hồi quy
beta_0 = model.intercept_  # Hệ số chặn
beta_1 = model.coef_[0]    # Hệ số dốc

# Đánh giá mô hình
r2 = r2_score(y, y_pred)  # Hệ số xác định R^2
mse = mean_squared_error(y, y_pred)  # Sai số bình phương trung bình (MSE)

# Hiển thị kết quả
print(f"Phương trình hồi quy: y = {beta_0:.2f} + {beta_1:.2f}x")
print(f"Hệ số chặn (beta_0): {beta_0}")
print(f"Hệ số dốc (beta_1): {beta_1}")
print(f"Hệ số xác định (R^2): {r2}")
print(f"Sai số bình phương trung bình (MSE): {mse}")

# Vẽ biểu đồ đánh giá mô hình
plt.scatter(X, y, color="blue", label="Dữ liệu thực tế")
plt.plot(X, y_pred, color="red", label="Dự đoán (hồi quy)")
plt.title("Hồi quy tuyến tính: Điểm thi và Điểm tổng kết")
plt.xlabel("Điểm thi")
plt.ylabel("Điểm tổng kết")
plt.legend()
plt.grid(True)
plt.show()