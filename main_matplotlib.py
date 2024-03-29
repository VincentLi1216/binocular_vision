import gradio as gr
from util import divide_img, stereo_matching
import cv2
import numpy as np

def threshold_image(image, lower_bound, upper_bound):
    """
    處理灰階圖片，將不在指定上下界範圍內的像素值設置為0。

    參數:
    image -- 灰階圖片 (NumPy array)
    lower_bound -- 下界 (int)
    upper_bound -- 上界 (int)

    返回:
    處理後的圖片
    """
    # 複製圖片以避免修改原始圖片
    processed_image = np.copy(image)

    # 將不在範圍內的像素值設置為0
    processed_image[(image < lower_bound) | (image > upper_bound)] = 0

    return processed_image

def update_image(scale_value1, scale_value2):
    print(f"Slider 1: {scale_value1}, Slider 2: {scale_value2}")
    # 這裡的 'source-4/Explorer_HD720_SN27863180_19-40-24.png' 是預設圖片的路徑
    left_img, right_img = divide_img("source-4/Explorer_HD720_SN27863180_19-40-24.png")
    disparity = stereo_matching(left_img, right_img, resize_ratio=0.35, numDisparities=64, blockSize=5)
    disparity = threshold_image(disparity, scale_value1, scale_value2)
    # # heat map
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    disparity = cv2.cvtColor(disparity, cv2.COLOR_BGR2RGB)

    print(disparity.shape)
    print(disparity)

    return disparity

# 定義 Gradio 界面
iface = gr.Interface(
    fn=update_image,
    inputs=[gr.Slider(0, 255), gr.Slider(0, 255)],
    outputs="image",
    live=True
)

# 運行 Gradio 應用程序
iface.launch(debug=True)
