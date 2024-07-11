import numpy as np
import cv2
import os

# 文件路径设置
file_path = "data/rgb/strawberries_coffee_3x256x256.bmp"

output_rgb_binary_file  = "data/rgb/strawberries_coffee_3x256x256_u8.raw"
output_rgb_jpeg_file    = "data/rgb/strawberries_coffee_3x256x256.jpeg"
output_rgb_png_file     = "data/rgb/strawberries_coffee_3x256x256.png"

output_gray_bmp_file    = "data/gray/strawberries_coffee_1x256x256.bmp"
output_gray_png_file    = "data/gray/strawberries_coffee_1x256x256.png"
output_gray_jpeg_file   = "data/gray/strawberries_coffee_1x256x256.jpeg"
output_gray_binary_file = "data/gray/strawberries_coffee_1x256x256_u8.raw"

# 创建必要的目录
os.makedirs(os.path.dirname(output_rgb_binary_file), exist_ok=True)
os.makedirs(os.path.dirname(output_gray_binary_file), exist_ok=True)

# 读取 BMP 文件
image_rgb = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

# 将图像保存为 BMP 文件
cv2.imwrite(file_path, image_rgb)

# 将 RGB 图像数据转换为 NumPy 数组
image_rgb_np = np.array(image_rgb)

# 手动打开文件并写入 RGB 图像的二进制数据
with open(output_rgb_binary_file, 'wb') as f:
    f.write(image_rgb_np.tobytes())

# 保存 RGB 图像为 PNG 文件
cv2.imwrite(output_rgb_png_file, image_rgb)

# 保存 RGB 图像为 JPEG 文件
cv2.imwrite(output_rgb_jpeg_file, image_rgb)

# 将 RGB 图像转换为灰度图
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

# 保存灰度图像为 BMP 文件
cv2.imwrite(output_gray_bmp_file, image_gray)

# 保存灰度图像为 PNG 文件
cv2.imwrite(output_gray_png_file, image_gray)

# 保存灰度图像为 JPEG 文件
cv2.imwrite(output_gray_jpeg_file, image_gray)

# 将灰度图像数据转换为 NumPy 数组
image_gray_np = np.array(image_gray)

# 手动打开文件并写入灰度图像的二进制数据
with open(output_gray_binary_file, 'wb') as f:
    f.write(image_gray_np.tobytes())
