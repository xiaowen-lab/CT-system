import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor  # 用于加载图像分类模型
import timm
import requests
from io import BytesIO


# 设置页面配置，包括页面标题和图标
st.set_page_config(page_title="Brain Stroke CT Image Diagnosis System", page_icon="https://raw.githubusercontent.com/xiaowen-lab/CT-system/main/icon.jpg")

# Hugging Face 模型路径
model_url = "https://huggingface.co/woAlex666/CT-VIT/resolve/main/vit_model.pth"

# 加载模型的权重
def load_model_from_huggingface(model_url):
    # 下载模型文件
    response = requests.get(model_url)
    if response.status_code == 200:
        model_weights = BytesIO(response.content)
        # 加载到PyTorch模型
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        num_classes = 2  # 二分类
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        checkpoint = torch.load(model_weights, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        return model
    else:
        st.error("Failed to load model from Hugging Face.")
        return None

# 加载模型
model = load_model_from_huggingface(model_url)

if model:
    model.eval()  # 设置为评估模式

    # 强制使用 CPU
    device = torch.device("cpu")
    model.to(device)

    # 数据预处理（保持与原始代码一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 类别映射
    class_names = ['Normal', 'Stroke']

    # 语言内容定义（此部分代码无需更改）

    content = {
        'en': {
            'title': 'Brain Stroke CT Image Intelligent Diagnosis System',
            'language_select': "Select Language / 选择语言",
            'upload_msg': "Choose an image",
            'predict_btn': "Start Prediction",
            'result_msg': "Prediction Result",
            'diagnosis_msg': "Diagnosis",
            'probability_msg': "Prediction Probability",
            'video_msg': " Video Tutorial",
            'video_tutorial_url': 'https://raw.githubusercontent.com/xiaowen-lab/CT-system/main/video.mp4',
            'upload_prompt': 'Upload Brain Stroke CT Image',
            'upload_file_hint': 'Drag and drop a file here or click to select a file.',
            'system_description': """
                This system is based on the ViT (Vision Transformer) network structure, optimized and fine-tuned using the timm library, specifically designed for intelligent diagnosis of brain stroke CT images. 
                The model automatically extracts features from CT images using deep learning techniques, and leverages the self-attention mechanism of the Transformer architecture to model global features, improving the accuracy of stroke diagnosis.
                Users only need to upload a brain stroke CT image, click the prediction button, and the system will automatically analyze the image, output the prediction results and disease diagnosis, and provide the corresponding confidence probability. 
                This system enables fast and high-precision stroke diagnosis support, assisting doctors in making timely and accurate clinical decisions.
            """,
            'navigate': 'Navigate'
        },
        'cn': {
            'title': '脑卒中CT图像智能诊断系统',
            'language_select': "选择语言 / Select Language",
            'upload_msg': "选择一张脑卒中CT图像",
            'predict_btn': "开始预测",
            'result_msg': "预测结果",
            'diagnosis_msg': "诊断结果",
            'probability_msg': "预测概率",
            'video_msg': " 视频教程",
            'video_tutorial_url': 'https://raw.githubusercontent.com/xiaowen-lab/CT-system/main/video.mp4',
            'upload_prompt': '上传脑卒中CT图像',
            'upload_file_hint': '拖拽文件到此处或点击选择文件。',
            'system_description': """
                本系统基于ViT（Vision Transformer）网络结构，结合timm库进行优化和精调，专门设计用于脑卒中CT图像的智能诊断。该模型通过深度学习技术，从CT图像中自动提取特征，并利用Transformer架构的自注意力机制进行全局特征建模，提升了对脑卒中诊断的准确性。
                用户只需上传一张脑卒中CT图像，点击预测按钮，系统将自动对图像进行分析，输出预测结果及疾病诊断，并提供相应的置信度概率。该系统能够实现快速且高精度的脑卒中诊断支持，辅助医生做出及时且准确的临床决策。
            """,
            'navigate': '导航'
        }
    }

    # 语言选择
    lang = st.selectbox(content['en']['language_select'], ('English', '中文'))

    # 当前语言设置
    current_lang = 'en' if lang == 'English' else 'cn'

    # 导航栏
    page = st.sidebar.radio(
        content[current_lang]['navigate'],
        [content[current_lang]['title'], content[current_lang]['video_msg']]
    )

    # 根据语言选择渲染页面内容
    if page == content[current_lang]['title']:
        # 显示系统名称
        st.title(content[current_lang]['title'])

        # 显示系统描述信息
        st.write(content[current_lang]['system_description'])

        # 显示系统Logo在系统介绍下方
        st.image('https://github.com/xiaowen-lab/Stroke-CT-Diagnosis-Prediction-System/blob/main/stroke-logo.jpg', use_container_width=True)

        # 显示上传框提示
        st.write(content[current_lang]['upload_prompt'])

        # 手动实现文件上传提示，避免显示默认英文提示
        upload_file_hint = content[current_lang]['upload_file_hint']
        uploaded_file = st.file_uploader(upload_file_hint, type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            image = image.convert("RGB")  # 确保是RGB格式
            st.image(image, caption='Uploaded Image', use_container_width=True)

            # 显示预测按钮
            predict_button = st.button(content[current_lang]['predict_btn'])

            if predict_button:
                # 转换为模型输入格式
                image_tensor = transform(image).unsqueeze(0).to(device)

                # 模型推理
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = class_names[predicted.item()]
                    probability = torch.softmax(outputs, dim=1)[0, predicted.item()].item()

                # 将概率转换为百分比
                probability_percentage = probability * 100

                # 结果显示框
                with st.container():
                    st.markdown(f"### {content[current_lang]['result_msg']}")
                    st.write(f"**{content[current_lang]['diagnosis_msg']}**：{predicted_class}")
                    st.write(f"**{content[current_lang]['probability_msg']}**：{probability_percentage:.2f}%")

    elif page == content[current_lang]['video_msg']:
        # 显示视频教程页面
        st.title(content[current_lang]['video_msg'])
        st.video(content[current_lang]['video_tutorial_url'], start_time=0)

