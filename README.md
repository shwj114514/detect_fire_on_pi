#　基于YOLOV5网络的树莓派火灾报警系统
## 模型架构
所用器件：温度传感器　湿度传感器　蜂鸣器　
## 配置环境
1. 查看树莓派系统或者是 Linux 系统是多少位系统的命令
getconf LONG_BIT
`uname -a`

2. 下载对应版本的torch
```
pip3 install torch-1.8.1-cp37-cp37m-linux_armv7l.whl
pip3 install torchvision-0.9.1-cp37-cp37m-linux_armv7l.whl 
```
3. 运行test/test_torch.py 和test/test_camera.py 确保torch和camera正常
4. 运行test.py确保运行正常
## run
详细代码在main.py中
## 更改模型
`NMS 输出list of detections, on (n,6) tensor per image [xyxy, conf, cls]`

## 参考代码
1. [yolo](https://github.com/ultralytics/yolov5)
2. [火灾检测](https://blog.csdn.net/weixin_63866037/article/details/128944444)

## 以下是yolo原代码的操作
### 语义分割测试
`python segment/predict.py --weights yolov5m-seg.pt --data data/images/bus.jpg`
### 图像分类测试
python detect.py --weights yolov5m.pt --img 640 --conf 0.25 --source data/images

### 火灾检测测试
python detect.py --weights yolo_fire.pt --img 640 --conf 0.25 --source data/fire_imgs
