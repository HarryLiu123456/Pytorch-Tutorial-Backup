#1.1手写数字识别

import model
import torch

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datasets
import model

def show():
    with torch.no_grad():
        img_path = './digital_recognition/pictures/test.png'
        image_origin = Image.open(img_path)
        image_processed = image_origin.convert("L")

        image_test = datasets.test_transform(image_processed)
        image_test = image_test.unsqueeze(dim=0)    #在第0个位置加一个维度, 长度为1
        #print(image_test.shape)
        model.net.eval()
        label_test = model.net(image_test)
        _,predicted = torch.max(label_test,1)
        print(predicted)

show()

epochs = 10
i = 0
while i<epochs:
    model.train(i)
    i+=1

show()
model.eval_3()
