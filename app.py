import streamlit as st
from ultralytics import YOLO
from PIL import Image
from util import classify, set_background

set_background('./Project/bg/bg.jpg')

st.title('Pneumonia classification')

st.header('Please upload a chest X-ray image')

file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

model = YOLO('./runs/classify/train6/weights/best.pt')

with open('./Project/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width = True)

    class_name, conf_score = classify(image, model, class_names)

    st.write(f'## {class_name}')
    st.write(f'## score: {conf_score * 100}%')