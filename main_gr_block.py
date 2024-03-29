import gradio as gr
from util import divide_img, stereo_matching, apply_jet_colormap_and_overlay
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


def update_image(
    scale_value1,
    scale_value2,
    ip_resize_ratio,
    ip_numDisparities,
    ip_blockSize,
    ip_kernel_size,
    ip_blur_size,
):
    print(f"Slider 1: {scale_value1}, Slider 2: {scale_value2}")
    # 這裡的 'source-4/Explorer_HD720_SN27863180_19-40-24.png' 是預設圖片的路徑
    left_img, right_img = divide_img("source-5/Explorer_HD720_SN27863180_15-41-53.png")
    disparity = stereo_matching(
        left_img,
        right_img,
        resize_ratio=ip_resize_ratio,
        numDisparities=ip_numDisparities,
        blockSize=ip_blockSize,
        kernel_size=ip_kernel_size,
        blur_size=ip_blur_size,
        to_show=False,
    )
    disparity = threshold_image(disparity, scale_value1, scale_value2)
    original_disparity = disparity.copy()
    # # heat map
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    disparity = cv2.cvtColor(disparity, cv2.COLOR_BGR2RGB)

    left_img, right_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(
        right_img, cv2.COLOR_BGR2RGB
    )
    # print(disparity.shape)
    # print(disparity)

    left_img_RGB = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    overlay_img = apply_jet_colormap_and_overlay(left_img_RGB, original_disparity)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    return left_img, right_img, disparity, overlay_img


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            slider1 = gr.Slider(0, 255, value=0, label="Lower Bound", interactive=True)
            slider2 = gr.Slider(
                0, 255, value=255, label="Upper Bound", interactive=True
            )
        with gr.Row():
            slider3 = gr.Slider(
                0, 1, value=0.35, label="resize_ratio", interactive=True
            )
            slider4 = gr.Slider(
                0, 128, value=64, label="numDisparities", interactive=True
            )
            slider5 = gr.Slider(0, 20, value=5, label="blockSize", interactive=True)
        with gr.Row():
            slider6 = gr.Slider(0, 20, value=13, label="kernel_size", interactive=True)
            slider7 = gr.Slider(0, 20, value=5, label="blur_size", interactive=True)
        with gr.Row():
            left_img = gr.Image(label="left image")
            right_img = gr.Image(label="right image")
        with gr.Row():
            output_image = gr.Image(label="Processed Image")
            overlay_img = gr.Image(label="Overlay Image")
        inputs_ = [slider1, slider2, slider3, slider4, slider5, slider6, slider7]
        outputs_ = [left_img, right_img, output_image, overlay_img]

        slider1.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider2.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider3.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider4.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider5.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider6.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )
        slider7.change(
            fn=update_image,
            inputs=inputs_,
            outputs=outputs_,
        )

    demo.launch(debug=True)


if __name__ == "__main__":
    main()
