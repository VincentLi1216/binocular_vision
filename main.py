import cv2
import numpy as np
import matplotlib.pyplot as plt


# list all pngs under source folder
def list_all_pngs(path):
    import os

    pngs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                pngs.append(os.path.join(root, file))
    return pngs


def apply_jet_colormap_and_overlay(color_image, grayscale_image):
    # Check if images are loaded properly
    if color_image is None or grayscale_image is None:
        raise ValueError("One or both images are not loaded properly.")
    

    # Resize grayscale image to match color image dimensions
    grayscale_image_resized = cv2.resize(
        grayscale_image, (color_image.shape[1], color_image.shape[0])
    )

    # 如果是浮點數影像，先轉換為8位元
    if grayscale_image_resized.dtype != np.uint8:
        grayscale_image_resized = cv2.normalize(grayscale_image_resized, None, 0, 255, cv2.NORM_MINMAX)
        grayscale_image_resized = cv2.convertScaleAbs(grayscale_image_resized)

    # print(grayscale_image)

    # Apply jet colormap to the grayscale image
    color_mapped_image = cv2.applyColorMap(grayscale_image_resized, cv2.COLORMAP_JET)

    # Overlay the color mapped image on the color image
    overlaid_image = cv2.addWeighted(color_image, 0.5, color_mapped_image, 0.5, 0)

    return overlaid_image


# divide img into 2 from the certer
def divide_img(img_path):
    # read as gray scale
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    left_img = img[0:height, 0 : center[0]]
    right_img = img[0:height, center[0] : width]
    return left_img, right_img


def stereo_matching(left_img, right_img):
    print(left_img.shape)
    resize_ratio = 0.2
    resize_x = int(2208 * resize_ratio)
    resize_y = int(1242 * resize_ratio)
    print(resize_x, resize_y)
    # resize image
    left_img = cv2.resize(left_img, (resize_x, resize_y))
    right_img = cv2.resize(right_img, (resize_x, resize_y))

    # 使用 StereoBM 或 StereoSGBM 來計算視差
    # 這裡使用 StereoBM 作為示例
    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=31)
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right)

    # generate overly image
    overlay_img = apply_jet_colormap_and_overlay(left_img, disparity)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    # show disparity and two images with matplotlib with subplots
    plt.figure(figsize=(40, 10))  # Increase the figure size to make it larger

    plt.subplot(221)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    plt.imshow(left_img)
    plt.title("Left Image")

    plt.subplot(222)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    plt.imshow(right_img, cmap="gray")
    plt.title("Right Image")

    plt.subplot(223)
    plt.imshow(disparity, cmap="jet")
    plt.title("Disparity Map")

    plt.subplot(224)
    plt.imshow(overlay_img)
    plt.title("Overlay Image")
    plt.show()

    # 視差圖可能需要一些後處理來改善結果
    # 例如使用濾波器或者對視差值進行範圍限定

    # 將視差圖轉換為深度圖
    # 這需要相機的焦距 (f) 和基線距離 (B)
    # 深度 Z 可以通過 Z = f * B / disparity 計算
    # 這裡假設焦距和基線距離已知
    f = 1.0  # 替換為實際的焦距值
    B = 1.0  # 替換為實際的基線距離

    # 計算深度圖，避免除以零
    with np.errstate(divide="ignore"):
        depth = f * B / disparity

    # 深度圖的值可能需要根據你的具體應用進行調整和範圍限定

    # 由於我無法直接展示生成的深度圖，這段代碼應作為指導性範例
    # 你需要根據自己的具體情況進行調整和優化


if __name__ == "__main__":
    file_path = list_all_pngs("./source-4")
    for img in file_path:
        # print(img)
        left_img, right_img = divide_img(img)
        stereo_matching(left_img, right_img)
