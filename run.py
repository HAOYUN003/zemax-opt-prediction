import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as job
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_excel("kaggle.xlsx")
X = df[['R1', 'R2', '透镜厚度', '像面位置']]
y = df['RMS光斑']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("===== 光学系统机器学习预测结果 =====")
print(f"RMSE误差: {np.sqrt(mean_squared_error(y_test, y_pred)):.5f}")
print(f"R²准确率: {r2_score(y_test, y_pred):.4f}")

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.xlabel('真实RMS光斑')
plt.ylabel('预测RMS光斑')
plt.title('真实值 vs 预测值')
plt.show()

job.dump(model,"opt_model.pkl")
job.dump(scaler,"opt_scaler.pkl")
print("\nModel saved successfully")