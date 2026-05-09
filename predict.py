import joblib
import numpy as np

model=joblib.load("opt_model.pkl")
scaler=joblib.load("opt_scaler.pkl")

print("输入：R1 R2 透镜厚度 像面位置，空格分隔，q退出")
while True:
    s=input("\n请输入：")
    if s.lower()=="q":break
    try:
        arr=list(map(float,s.split()))
        if len(arr)!=4:
            print("需4个参数")
            continue
        res=model.predict(scaler.transform([arr]))[0]
        print(f"预测RMS光斑：{res:.6f}")
    except:
        print("输入错误")