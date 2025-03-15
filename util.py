import base64
import streamlit as st

def set_background(image_file):
    with open(image_file, 'rb') as f:
        img_data = f.read()
    
    b64_encoded = base64.b64encode(img_data).decode()

    style = f"""
        <style>
        .stApp{{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """

    st.markdown(style, unsafe_allow_html = True)

def classify(image, model, class_names):

    result = model(image)

    class_id = int(result[0].probs.top1)
    confidence = round(float(result[0].probs.top1conf), 3)

    class_name = class_names[class_id]

    return class_name, confidence