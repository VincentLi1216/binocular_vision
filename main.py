import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from util import divide_img, stereo_matching
import tkinter as tk

def cv2img_to_tkimg(cv2_img):
    # 如果圖片數據類型是 CV_16S，將其轉換為 CV_8U
    if cv2_img.dtype == 'int16':
        cv2_img = cv2.convertScaleAbs(cv2_img)

    # 判斷是否是灰階圖片
    if len(cv2_img.shape) == 2:
        # 將灰階圖片轉換為三通道
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    else:
        # 將 BGR 格式轉換為 RGB 格式
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # 轉換為 PIL 圖片
    pil_img = Image.fromarray(cv2_img)

    # 將 PIL 圖片轉換為 Tkinter 圖片
    tk_img = ImageTk.PhotoImage(image=pil_img)
    return tk_img


def update_image(scale_value1, scale_value2):
    # 這裡應該是更新圖片的邏輯
    # 例如，使用 scale_value1 和 scale_value2 調整圖片的某些屬性
    print(f"Slider 1: {scale_value1}, Slider 2: {scale_value2}")
    # 更新圖片顯示...
    left_img, right_img = divide_img("source-4/Explorer_HD720_SN27863180_19-40-24.png")
    updated_image = stereo_matching(left_img, right_img, resize_ratio=0.35, numDisparities=64, blockSize=5)
    # cv2.imshow("Disparity", updated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    updated_image = cv2img_to_tkimg(left_img)
    image_label.config(image=updated_image)
    
def slider_changed(event):
    scale_value1 = slider1.get()
    scale_value2 = slider2.get()
    update_image(scale_value1, scale_value2)

# 創建主窗口
root = tk.Tk()
root.title("圖片調整器")

# 創建滑動條
slider1 = ttk.Scale(root, from_=0, to=100, orient='horizontal', command=slider_changed)
slider1.pack()

slider2 = ttk.Scale(root, from_=0, to=100, orient='horizontal', command=slider_changed)
slider2.pack()

# 創建顯示圖片的標籤
image_label = tk.Label(root)
image_label.pack()

# 開始主事件循環
root.mainloop()
