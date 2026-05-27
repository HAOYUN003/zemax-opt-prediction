import cv2
import numpy as np
import os

# 你的照片文件夹（这里只需要改这1行路径）
folder_path = "光斑照片/"

# 遍历文件夹里所有图片
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".bmp")):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, 0)
        
        # 跳过空图
        if img is None:
            continue

        # 计算重心 & RMS
        y, x = np.indices(img.shape)
        total = img.sum()
        cx = (x * img).sum() / total
        cy = (y * img).sum() / total
        rms_x = np.sqrt( ((x - cx)**2 * img).sum() / total )
        rms_y = np.sqrt( ((y - cy)**2 * img).sum() / total )

        print(f"图片：{filename}")
        print(f"重心 cx={cx:.2f}, cy={cy:.2f}")
        print(f"RMS_x={rms_x:.2f}, RMS_y={rms_y:.2f}\n")