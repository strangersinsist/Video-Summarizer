import os
import sys
from collections import Counter
from model import Model
import cv2 as cv
import imageio
import numpy as np
import streamlit as st
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2
from wordcloud import WordCloud
from prompt import Prompt

# Load EdgeSAM Model
def load_segment_model():
    sys.path.append("tools/EdgeSAM")
    from edge_sam import sam_model_registry, SamPredictor

    sam_checkpoint = "weight/edge_sam.pth"
    model_type = "edge_sam"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor, device


# YOLO Model Loading
@st.cache_resource
def load_detection_model():
    model = torch.hub.load('tools/yolov5', 'custom', path='weight/yolov5l.pt', source='local')
    return model


# Function for Predicting Segmentation
def predict(predictor, points, labels, img_shape):
    w_scale = img_shape[1] / 640
    h_scale = img_shape[0] / 480
    args = {'num_multimask_outputs': 4, 'use_stability_score': True}
    masks, scores, _ = predictor.predict(
        point_coords=(np.array(points) * [w_scale, h_scale]).astype(int) if len(points) else None,
        point_labels=np.array(labels) if len(points) else None,
        **args
    )
    best_mask = masks[np.argmax(scores)]
    return best_mask


# Process Image for Segmentation
def process_image(predictor, img, points, labels, temp_point=None, temp_label=None):
    orig_img = deepcopy(img)
    viz_size = (640, 480)
    img = cv.resize(orig_img, viz_size)
    predictor.set_image(orig_img)

    def show_mask(mask, img):
        color = np.array([50 / 255, 144 / 255, 255 / 255])
        segment = img[mask, :].astype(float) * color
        img[mask, :] = segment.astype(int)

    mask = None
    if points:
        mask = predict(predictor, points, labels, orig_img.shape)
        show_mask(mask, orig_img)

    img = cv.resize(orig_img, viz_size)
    draw_points(img, points, labels, temp_point, temp_label)

    if mask is None:
        mask = np.zeros(orig_img.shape[:2], dtype=bool)

    return img, mask


def draw_points(img, points, labels, temp_point=None, temp_label=None):
    for i in range(len(points)):
        point = points[i]
        point = (int(point[0]), int(point[1]))
        color = (0, 255, 0) if labels[i] else (0, 0, 0)
        cv.circle(img, tuple(point), 5, color, -1)

    if temp_point is not None and len(temp_point) == 2:
        temp_point = (int(temp_point[0]), int(temp_point[1]))
        temp_color = (255, 0, 0) if temp_label else (0, 0, 255)
        cv.circle(img, tuple(temp_point), 5, temp_color, -1)


class WordCloudGenerator:
    def __init__(self):
        self.model_instance = Model()

    def generate_wordcloud(self, detected_object, width=800, height=500):
        prompt = Prompt.prompt1(ID='wordcloud2')

        ai_generated_text = self.model_instance.openai_chatgpt(detected_object, prompt)
        if isinstance(ai_generated_text, tuple):
            st.error(ai_generated_text[0])
            return None

        words_list = ai_generated_text.strip().split(', ')

        # 将关键词随机重复，确保足够数量的关键词填满蒙版
        word_frequencies = Counter(words_list)
        repeated_words = []
        for word, count in word_frequencies.items():
            # 每个词重复 2 到 4 次
            repeated_words.extend([word] * np.random.randint(2, 5))

        # 重新生成的词列表
        words = ' '.join(repeated_words)
        # 从文件中加载蒙版图像
        mask_image = imageio.v3.imread('temp_segmented_image.png')

        if mask_image is None:
            st.error("Mask image not found.")
            return None

        # 应用形态学操作使得轮廓更加平滑
        kernel = np.ones((5, 5), np.uint8)
        mask_processed = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)

        # 使用处理后的蒙版生成词云
        wordcloud1 = WordCloud(
            width=width,
            height=height,
            background_color='white',  # 背景为白色
            mask=mask_processed,  # 使用填充后的蒙版
            max_words=500,  # 增加最大词数量
            contour_width=2, #增加图像边框
            contour_color='blue',
        ).generate(words)

        st.session_state.wordcloud1 = wordcloud1
        return wordcloud1


# 调用侧边栏图像处理功能
def sidebar_image_processing(img):
    # 检查传入的图片是否与之前的不同
    if 'previous_image' not in st.session_state or not np.array_equal(st.session_state['previous_image'],
                                                                      np.array(img)):
        # 保存新图像
        st.session_state['previous_image'] = np.array(img)

        # 重置所有相关状态变量，包括清空点和标签
        st.session_state['points'] = []
        st.session_state['labels'] = []
        st.session_state['temp_point'] = None
        st.session_state['temp_label'] = None

        # 重置分割图像和蒙版
        if 'segmented_region' in st.session_state:
            del st.session_state['segmented_region']
        if 'clean_segmented_region' in st.session_state:
            del st.session_state['clean_segmented_region']
        if 'mask_shape' in st.session_state:
            del st.session_state['mask_shape']

        # 重置词云
        st.session_state.wordcloud1 = None

    predictor, device = load_segment_model()
    detection_model = load_detection_model()

    st.sidebar.title("Image Stick")

    # 使用传入的img
    orig_img = np.array(img)

    # 变量控制是否显示处理后的图像和掩码
    show_processed_img = not ('segmented_region' in st.session_state or 'clean_segmented_region' in st.session_state)

    # 创建 Streamlit canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        background_image=img,  # 使用传入的img作为背景
        update_streamlit=True,
        height=480,
        width=640,
        drawing_mode="point",
        key="canvas",
    )

    # 检查是否有新的点被添加到 canvas 上
    if canvas_result.json_data is not None:
        # 清空之前的点和标签，重新收集画布上的所有点
        st.session_state['points'] = []
        st.session_state['labels'] = []
        for obj in canvas_result.json_data["objects"]:
            point = obj["left"], obj["top"]
            st.session_state['points'].append(point)
            st.session_state['labels'].append(1)  # 假设所有点都标记为 1，你可以根据需求修改
        show_processed_img = True


    # 保存分割区域
    if st.sidebar.button("Save Segment"):
        if st.session_state['points']:
            # 重新生成掩码
            _, mask = process_image(predictor, orig_img, st.session_state['points'], st.session_state['labels'])

            rgba = cv.cvtColor(orig_img, cv.COLOR_RGB2RGBA)
            output_img = np.zeros_like(rgba)
            row_coords, col_coords = np.nonzero(mask)
            output_img[row_coords, col_coords] = rgba[row_coords, col_coords]

            rmin = min(row_coords)
            rmax = max(row_coords)
            cmin = min(col_coords)
            cmax = max(col_coords)

            segmented_region = output_img[rmin:rmax + 1, cmin:cmax + 1]

            # 生成白色背景的分割图像
            background = np.ones_like(rgba) * 255  # 白色背景
            background[row_coords, col_coords] = rgba[row_coords, col_coords]  # 替换为分割的部分

            temp_filename = "temp_segmented_image.png"
            cv.imwrite(temp_filename, cv.cvtColor(background, cv.COLOR_RGBA2BGR))  # 保存图像

            # 保存分割图像和蒙版到 session state
            st.session_state['segmented_region'] = segmented_region
            st.session_state['mask_shape'] = temp_filename  # 保存文件名以备后续使用
            st.session_state['clean_segmented_region'] = segmented_region

            st.sidebar.image(segmented_region, caption="Segmented Region", use_column_width=True)
            show_processed_img = False

    # 在对象检测和生成词云的逻辑中使用
    if 'clean_segmented_region' in st.session_state:
        if st.sidebar.button("Detect Objects and Generate Word Cloud"):
            show_processed_img = False
            st.sidebar.image('temp_segmented_image.png', caption="Segmented Region", use_column_width=True)
            results = detection_model(st.session_state['mask_shape'])  # 使用保存的图像文件进行对象检测
            if results.xyxy[0].size(0) > 0:
                highest_confidence = -1
                best_label = None
                for result in results.xyxy[0]:
                    confidence = result[4].item()
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        class_id = int(result[5].item())
                        best_label = detection_model.names[class_id]

                if best_label:
                    st.sidebar.write(f"Detected Object: {best_label}")
                    wordcloud_generator = WordCloudGenerator()
                    wordcloud_image = wordcloud_generator.generate_wordcloud(best_label)  # 使用保存的图片作为mask
                    if wordcloud_image:
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud_image, interpolation='bilinear')
                        plt.axis('off')
                        st.sidebar.pyplot(plt)
                        os.remove('temp_segmented_image.png')
            else:
                st.sidebar.write("No objects detected.")

            # 处理并显示原图像和掩码，仅在未点击"Save Segment"或"Detect Objects and Generate Word Cloud"时执行
    if show_processed_img:
        processed_img, _ = process_image(predictor, orig_img, st.session_state['points'],
                                         st.session_state['labels'],
                                         st.session_state['temp_point'], st.session_state['temp_label'])
        st.sidebar.image(processed_img, use_column_width=True)
