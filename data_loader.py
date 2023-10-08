import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import rotate
from skimage import exposure
from utils import brightness,horizontal_shift,vertical_shift,zoom
from config import *

def batch_generator(batch_size, gen_x): 
    batch_features = np.zeros((batch_size,256,256,3))
    batch_labels = np.zeros((batch_size,256,256,3)) 
    while True:
        for i in range(batch_size):
            batch_features[i] , batch_labels[i] = next(gen_x)
        yield batch_features, batch_labels

def generate_data(filelist, img_path, mask_path, gen_type = "train"):
    while True:
        for i in filelist:
            X_train = cv2.imread(img_path + i, cv2.IMREAD_COLOR )
            X_train = cv2.resize(X_train, (256,256), interpolation= cv2.INTER_LINEAR )
#             i = i.split(".")[0] + ".png"
            i = "ann" + i.split(".")[0][3:] + ".png"
            y_mask = cv2.imread(mask_path + i, cv2.IMREAD_COLOR)
            y_mask = cv2.resize(y_mask, (256,256), interpolation= cv2.IMREAD_GRAYSCALE)
            _,y_mask = cv2.threshold(y_mask, 127, 255, cv2.THRESH_BINARY)
            y_train = (y_mask/255).astype(int)
            if gen_type == "train":
                # returns a random integer used to select augmentataion techniques for a given sample
                augment_num = np.random.randint(0,9)
                #augment_num = 0 
                if augment_num == 0:
                    # do nothing
                    X_train = X_train
                elif augment_num == 1:
                    #random noise
                    X_train = X_train + np.random.rand(X_train.shape[0], X_train.shape[1], X_train.shape[2])*np.random.randint(-100,100)
                elif augment_num == 2:
                    X_train = cv2.GaussianBlur(X_train,(random.randrange(1,50,2),random.randrange(1,50,2)), 0)
                elif augment_num == 3:
                    rot = np.random.randint(-45,45)
                    X_train = rotate(X_train,rot, reshape=False)
                    y_train = rotate(y_train,rot, reshape=False)
                elif augment_num == 4:
                    X_train = brightness(X_train,0.5,3)
                elif augment_num == 5:
                    X_train = np.fliplr(X_train)
                    y_train = np.fliplr(y_train)
                elif augment_num == 6:
                    X_train = np.flipud(X_train)
                    y_train = np.flipud(y_train)
                elif augment_num == 7:
                    hshift = round(random.uniform(0.1, 0.3),3)
                    X_train, y_train = horizontal_shift(X_train, y_train, hshift)
                elif augment_num == 8:
                    vshift = round(random.uniform(0.1, 0.3),3)
                    X_train, y_train = vertical_shift(X_train, y_train, vshift)
                elif augment_num == 9:
                    zoom_rate = round(random.uniform(0.8, 0.95),3)
                    X_train, y_train = zoom(X_train, y_train, zoom_rate)
                elif augment_num == 10:
                    #contrast
                    X_train = exposure.equalize_adapthist(X_train.astype(int), clip_limit=0.03)  
                elif augment_num == 11:
                    #contrast
                    X_train = exposure.equalize_hist(X_train.astype(int))  
            yield X_train, y_train


def generate_data_pred(filelist, img_path, mask_path, gen_type = "train"):
    while True:
        for i in filelist:
            original_img = cv2.imread(img_path + i, cv2.IMREAD_COLOR )
            X_train = cv2.resize(original_img, (256,256), interpolation= cv2.INTER_LINEAR )
            if gen_type == "train":
                X_train = X_train * np.random.choice([1,1,1,np.random.rand(256, 256,3)])
            i = "ann" + i.split(".")[0][3:] + ".png"
            original_mask = cv2.imread(mask_path + i, cv2.IMREAD_COLOR)
            y_mask = cv2.resize(original_mask, (256,256), interpolation= cv2.IMREAD_GRAYSCALE)
            _,y_mask = cv2.threshold(y_mask, 127, 255, cv2.THRESH_BINARY)
            y_mask = (y_mask/255).astype(int)
            yield original_img, original_mask, X_train, y_mask