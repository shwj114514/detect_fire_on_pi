def test_cv2():
    import cv2
    print(cv2.__version__)
    img=cv2.imread('a.jpg',0)
    print(img.shape)


def test_camera():
    import cv2  # 导入库

    cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    while True:
        ret, frame = cap.read()
        # frame读取cap的图像数据，返回ret，读取成功返回true,失败返回flase
        if ret:
            cv2.imshow('video', frame)  # 读取成功，显示窗口名为'video'的摄像头图像
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下键盘'q'键退出窗口
            break

    cv2.destroyAllWindows()  # 关闭窗口
    cap.release()  # 关闭摄像头数据读取

def test_camera2(): # test_camera1的imshow有问题
    import cv2  # 导入库

    cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    while True:
        ret, img = cap.read() #  (480, 640, 3)
        # frame读取cap的图像数据，返回ret，读取成功返回true,失败返回flase
        # img = img.permute(0, 3, 1, 2)
        print(img.shape) # 送入yolo的shape

    cv2.destroyAllWindows()  # 关闭窗口
    cap.release()  # 关闭摄像头数据读取

def test_camera3():
    import cv2  # 导入库
    import torch

    cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    while True:
        ret, img = cap.read() #  (480, 640, 3)
        im = torch.from_numpy(img)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        img = im.permute(0, 3, 1, 2)
        print(img.shape)

    cv2.destroyAllWindows()  # 关闭窗口
    cap.release()  # 关闭摄像头数据读取

def test_save():
    import numpy as np
    import cv2  # 导入库
    import time

    cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    ret, img = cap.read()
    # frame读取cap的图像数据，返回ret，读取成功返回true,失败返回flase
    print(img.shape)
    np.save("ceshi.npy",img)

    # cv2.imwrite("ceshi", img)
    cv2.destroyAllWindows()  # 关闭窗口
    cap.release()  # 关闭摄像头数据读取

if __name__ == '__main__':
    # test_cv2()
    test_camera3()