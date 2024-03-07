from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")
# model = YOLO("runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    result = model.predict(source="0", show=True, save=True)  # predict on an image
    # print(result)
