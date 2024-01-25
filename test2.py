from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    model.predict(source="images", save=True)  # predict on an image
