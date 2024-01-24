from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    # Use the model
    results = model.train(data="coco128.yaml", epochs=3)  # train the model
    results = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
