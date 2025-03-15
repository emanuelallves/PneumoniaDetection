from ultralytics import YOLO
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix

model = YOLO('./runs/classify/train6/weights/best.pt')

def load_images_and_labels(directory, label):
    images = []
    labels = []

    for file in os.listdir(directory):

        image_path = os.path.join(directory, file)

        if os.path.isfile(image_path) and image_path.endswith(('jpg', 'jpeg', 'png')):
            images.append(image_path)
            labels.append(label)

    return images, labels

def prediction(images):
    predicted_labels = []

    for image_path in images:

        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = model(image_rgb)
        predicted_label = int(result[0].probs.top1)
        predicted_labels.append(predicted_label)

    return predicted_labels

normal_folder = './Project/val/NORMAL'
pneumonia_folder = './Project/val/PNEUMONIA'


normal_images, normal_labels = load_images_and_labels(normal_folder, 0)
pneumonia_images, pneumonia_labels = load_images_and_labels(pneumonia_folder, 1)

images = normal_images + pneumonia_images
true_labels = normal_labels + pneumonia_labels

predicted_labels = prediction(images)

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize = (6, 5))
sns.heatmap(cm, annot = True,
            fmt = "d",
            cmap = "Blues",
            xticklabels = ["NORMAL", "PNEUMONIA"],
            yticklabels = ["NORMAL", "PNEUMONIA"])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.savefig('Project/results/Predict_CM.jpg', dpi = 300)