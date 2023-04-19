import cv2
import torch
im = cv2.imread('a.jpg')

im = torch.from_numpy(im).to('cuda')
im = im.float()
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim
im = im.permute(0, 1, 2, 3)
print(im.shape)