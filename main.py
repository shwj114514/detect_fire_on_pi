#####################################################
#!/usr/bin/env python
import LCD1602
import cv2
import threading
import time
import torch
import os

makerobo_ds18b20 = ''  # ds18b20 设备

def makerobo_setup():
	LCD1602.init(0x27, 1)	# init(slave address, background light)
	LCD1602.write(0, 0, 'Greetings!!')
	LCD1602.write(1, 1, 'WWW.HNZHIYU.CN')
	time.sleep(2)

	global makerobo_ds18b20  # 全局变量
	# 获取 ds18b20 地址
	for i in os.listdir('/sys/bus/w1/devices'):
		if i != 'w1_bus_master1':
			makerobo_ds18b20 = i       # ds18b20存放在ds18b20地址

	print("setup_finish!")

# 读取ds18b20地址数据
def makerobo_read():
	makerobo_location = '/sys/bus/w1/devices/' + makerobo_ds18b20 + '/w1_slave' # 保存ds18b20地址信息
	makerobo_tfile = open(makerobo_location)  # 打开ds18b20
	makerobo_text = makerobo_tfile.read()     # 读取到温度值
	makerobo_tfile.close()                    # 关闭读取
	secondline = makerobo_text.split("\n")[1] # 格式化处理
	temperaturedata = secondline.split(" ")[9]# 获取温度数据
	temperature = float(temperaturedata[2:])  # 去掉前两位
	temperature = temperature / 1000          # 去掉小数点
	return temperature                        # 返回温度值

# 循环函数
def thread_temperature():
	space = '     '
	greetings=''
	while True: # 每次循环的开始读取传感器的温度数值 拼接成字符串 传递给LCD 传感器上
		if makerobo_read() != None:  # 调用读取温度值，如果读到到温度值不为空
			# greetings ="Current temperature : %0.3f C" % makerobo_read()
			greetings = str(makerobo_read())
		greetings = space + greetings
		print(f"greetings={greetings}")

		tmp = greetings
		print(f"tmp={tmp}")
		LCD1602.write(0, 0, tmp) # 将数据显示到LCD上

def thread_detect(gap_time=180): #　选择gap时间以使得gap_time比yolo单帧检测的时间长
	cap = cv2.VideoCapture(0)  # 捕获摄像头设备图像数据存入cap。
	cap.set(3, 600)  # cap.set 摄像头参数设置
	cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
	cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

# 导入配置
	import os
	import sys
	from pathlib import Path
	FILE = Path(__file__).resolve()
	ROOT = FILE.parents[0]  # YOLOv5 root directory
	if str(ROOT) not in sys.path:
		sys.path.append(str(ROOT))  # add ROOT to PATH
	ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
	from models.common import DetectMultiBackend
	from utils.general import non_max_suppression
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



	while True:
		ret, frame = cap.read()
		# frame读取cap的图像数据，返回ret，读取成功返回true,失败返回flase
		# im = frame.float()
		img=torch.from_numpy(frame)
		img = img.float()
		img /= 255  # 0 - 255 to 0.0 - 1.0
		if len(img.shape) == 3:
			img = img[None]  # expand for batch dim

		img = img.permute(0, 3, 1, 2)
		# TODO [1,3,480,480] 送到yolo模型中得到结果
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

		time.sleep(gap_time)


	cv2.destroyAllWindows()  # 关闭窗口
	cap.release()  # 关闭摄像头数据读取

# 释放资源
def destroy():
	pass

# 程序入口
if __name__ == '__main__':
	try:
		makerobo_setup()  # 调用初始化程序
		t1 = threading.Thread(target=thread_temperature, name="fun_thread1", daemon=True)  # 创建thread1线程
		t2 = threading.Thread(target=thread_detect, name="fun_thread2", daemon=True)  # 创建thread2线程
		t1.start()  # 启动thread1线程
		t2.start()  # 启动thread2线程
		print("t1的线程名字是 %s" % t1.getName())  # 打印t1线程的线程名字
		print("t2的线程名字是 %s" % t2.getName())  # 打印t2线程的线程名字
		t1.join()  # 当前需要等待线程t1执行完毕后才能运行下一步
		t2.join()  # 当前需要等待线程t2执行完毕后才能运行下一步

	except KeyboardInterrupt: # 当按下Ctrl+C时，将执行destroy()子程序。
		destroy()             # 释放资源
		print("主线程执行完毕！")

