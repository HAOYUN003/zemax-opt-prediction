import numpy as np
import pandas as pd
import joblib

# ========== 1. 加载已经训练好的模型 & 归一化器 ==========
model = joblib.load("opt_model.pkl")
scaler = joblib.load("opt_scaler.pkl")

# ========== 2. 生成和训练集分布一致的实验参数 ==========
def generate_experiment_params(num=10):
    np.random.seed(42)  # 固定随机，结果可复现
    params = []
    for _ in range(num):
        # 严格匹配你kaggle数据集的参数范围
        R1 = np.random.uniform(20, 80)
        R2 = np.random.uniform(-80, -20)
        thickness = np.random.uniform(2, 12)
        image_pos = np.random.uniform(30, 200)
        params.append([R1, R2, thickness, image_pos])
    return np.array(params)

# 生成20组实验参数
new_params = generate_experiment_params(20)

# ========== 3. 预测RMS光斑（必须归一化，和训练流程一致） ==========
new_scaled = scaler.transform(new_params)
pred_rms = model.predict(new_scaled)

# ========== 4. 输出可直接做实验验证的表格 ==========
result = pd.DataFrame({
    "序号": range(1, len(new_params)+1),
    "R1(mm)": new_params[:,0].round(2),
    "R2(mm)": new_params[:,1].round(2),
    "透镜厚度(mm)": new_params[:,2].round(2),
    "像面位置(mm)": new_params[:,3].round(2),
    "模型预测RMS(μm)": pred_rms.round(3)
})

print("===== 可直接用于实验验证的参数+预测结果 =====")
print(result.to_string(index=False))

# 保存Excel并与实验室测真实光斑对比
result.to_excel("光学光斑_实验验证参数表.xlsx", index=False)
print("\n 已保存文件：光学光斑_实验验证参数表.xlsx")