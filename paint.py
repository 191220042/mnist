import tkinter as tk
from tkinter import messagebox

import cv2
import torch
from PIL import Image
import io
import torch.nn as nn
from torchvision.transforms import transforms
from MNIST.mnistGPU import CNN
# # 加载预训练的模型
model = torch.load('mnist_model.pt')
model = model.cpu()
model.eval()


# 设置转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 创建窗口
window = tk.Tk()
window.title("Digit Recognition")

# 创建画布
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack()

# 定义画笔状态
drawing = False
last_x = 0
last_y = 0

# 绘制数字
def paint(event):
    global drawing, last_x, last_y
    if drawing:
        canvas.create_line((last_x, last_y, event.x, event.y), width=20)
    last_x = event.x
    last_y = event.y

# 开始绘制
def start_paint(event):
    global drawing, last_x, last_y
    drawing = True
    last_x = event.x # 初始化 last_x
    last_y = event.y # 初始化 last_y


# 停止绘制
def stop_paint(event):
    global drawing
    drawing = False
    # 进行数字识别
    digit_image = canvas_to_image()
    prediction = recognize_digit(digit_image)
    messagebox.showinfo("Prediction", f"The digit is: {prediction}")

# 将画布转换为图像
def canvas_to_image():
    digit_image = canvas.postscript(colormode='gray')
    digit_image = Image.open(io.BytesIO(digit_image.encode('utf-8')))
    digit_image = digit_image.convert('L')  # 转换成灰度图片
    digit_image = digit_image.resize((28, 28))
    digit_image.show()
    digit_image = transform(digit_image)
    digit_image = digit_image.unsqueeze(0)


    return digit_image

# 数字识别
def recognize_digit(image):
    with torch.no_grad():
        output = model(image)


        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# 清空画布
def clear_canvas():
    canvas.delete("all")

# 创建清空按钮
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

# 绑定事件
canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", start_paint)
canvas.bind("<ButtonRelease-1>", stop_paint)

# 运行窗口程序
window.mainloop()
