import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-dysample.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='D://project//pycharm//pytorch//ultralytics-main//dataset1//data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                close_mosaic=10,
                workers=1,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )