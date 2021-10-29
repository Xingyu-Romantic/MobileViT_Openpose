import cv2
import matplotlib.pyplot as plt
import demo1.cv_demo1 as cv_demo1
import numpy as np


image_rgb = cv2.imread('./lena.jpg', cv2.IMREAD_UNCHANGED)

var1 = cv_demo1.test_rgb_to_gray(image_rgb)
print(var1.shape)
plt.figure('rgb-gray')
plt.imshow(var1, cmap=plt.gray())



plt.show()