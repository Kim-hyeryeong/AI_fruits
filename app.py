import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# VGG16 모델 로드 함수
@st.cache(allow_output_mutation=True)
def load_model():
    # Check if the file exists 
    model_path = "/Users/kimhyeryeong/Desktop/fruit_classifier3.h5" #경로
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop() # Stop execution if model file is not found
    
    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.resize((224, 224))  # VGG16 입력 크기
    image = np.array(image) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 신선도 예측 함수
def predict_freshness(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

# Streamlit UI
st.title("Fruit Freshness Checker with VGG16")
st.write("Upload an image of a fruit to check its freshness!")

# 파일 업로드 UI
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 신선도 예측
    with st.spinner("Analyzing..."):
        predictions = predict_freshness(image)
        freshness_score = predictions[0][0]  # VGG16 출력 결과

    # 결과 출력
    st.success("Analysis Complete!")
    st.write(f"Freshness Score: {freshness_score:.2f}")

    if freshness_score > 0.5:
        st.write("The fruit is likely **Fresh**!")
    else:
        st.write("The fruit may not be fresh.")