import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import non_max_suppression


def detect_single_img(img):
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    # Directories
    weights = 'yolo_fire.pt'  # model path or triton URL
    # Load model
    device = torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=False, fp16=False)
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    is_detect = False
    probability = -1
    for det in pred:
        if len(det):
            is_detect = True
    if is_detect:
        probability = torch.max(pred[0].T, dim=-1)[0][4]
    print("火灾的置信度为{}".format(probability))


def test_camera():
    import cv2  # 导入库
    import torch
    print(cv2.__version__)
    print(torch.__version__)

    cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    ret, img = cap.read() #  (480, 640, 3)
    im = torch.from_numpy(img)
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    img = im.permute(0, 3, 1, 2)
    print(img.shape)
    detect_single_img(img)

    cv2.destroyAllWindows()  # 关闭窗口
    cap.release()  # 关闭摄像头数据读取


if __name__ == '__main__':
    test_camera()