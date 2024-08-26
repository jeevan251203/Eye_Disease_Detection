from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    model.train(data="train.yaml", epochs=500)  # train the model
if __name__ == '__main__':
    main()

