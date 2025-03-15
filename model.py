from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(data = 'C:\Portfolio\ComputerVision\ImageClassification\pneumonia\Project\chest_xray',
            epochs = 10,
            imgsz = 64)