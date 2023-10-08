import numpy as np
import os
from sklearn.model_selection import KFold
import tensorflow as tf


file_list = np.asarray(os.listdir("Submission_segmentation/data/WCEBleedGen/WCEBleedGen/bleeding/Images"))
image_path = "Submission_segmentation/data/WCEBleedGen/WCEBleedGen/bleeding/Images/" 
mask_path = "Submission_segmentation/data/WCEBleedGen/WCEBleedGen/bleeding/Annotations/"

batchsize = 30
data_size = len(file_list)
num_epoch = 5
splits = 10
kf = KFold(n_splits=splits)
valsize = data_size // splits
trainsize = data_size - valsize

my_model = "efficientnetb1"
data_num = np.arange(data_size)

img_size = (256, 256)
num_classes = 3

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='min',
    min_delta=0.0001, cooldown=4, min_lr=0
)
print(file_list)