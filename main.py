import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image
import PIL.ImageOps
import os
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)

xtrain = xtrain/255.0
xtest = xtest/255.0

clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrain,ytrain)

def get_pred(image):
    impil = Image.open(image)
    image_bw = impil.convert('L')
    image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scaled = np.clip(image_bw_resize-min_pixel,0,255)
    max_pixel = np.max(image_bw_resize)
    image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred
