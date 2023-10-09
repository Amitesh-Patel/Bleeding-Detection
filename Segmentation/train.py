from sklearn.metrics import jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from metrics import *
from data_loader import *
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.framework()


validation_dice_original = np.zeros([valsize,splits])
validation_dice_resized = np.zeros([valsize,splits])
validation_jaccard_original = np.zeros([valsize,splits])
validation_jaccard_resized = np.zeros([valsize,splits])
cv_count = 0

all_history= []
for train_index, val_index in kf.split(data_num):

    #model = get_model(img_size, num_classes)
    model = sm.Unet(my_model, encoder_weights="imagenet", input_shape=( 256,256, 3), classes=3, activation='sigmoid')
    model.compile(optimizer='Adam', loss=jacard_coef_loss, metrics = [jacard_coef, dice_coef])
    history = model.fit(x=batch_generator(batchsize, generate_data(file_list[train_index], image_path, mask_path, gen_type = "train")), epochs=num_epoch, 
                            steps_per_epoch=(trainsize/batchsize), 
                            validation_steps=(valsize/batchsize),
                            validation_data=batch_generator(batchsize, generate_data(file_list[val_index], image_path, mask_path, gen_type = "val")), 
                            validation_freq=1, 
                            verbose = 1, 
                            callbacks=[reduce_lr],
                            )
    val_gen  = generate_data_pred(file_list[val_index], image_path, mask_path, gen_type = "val")
    for i in range(valsize):
        time_start = time.time()
        original_img, original_mask, X, y_true = next(val_gen)
        original_shape = original_img.shape
        y_pred = model.predict(np.expand_dims(X,0))
        _,y_pred_thr = cv2.threshold(y_pred[0,:,:,0]*255, 127, 255, cv2.THRESH_BINARY)
        y_pred = (y_pred_thr/255).astype(int)
        dice_resized = dice_score(y_true[:,:,0],y_pred)
        jaccard_resized = jaccard_score(y_true[:,:,0],y_pred, average="micro")
        
        y_pred_original = cv2.resize(y_pred.astype(float), (original_shape[1],original_shape[0]), interpolation= cv2.INTER_LINEAR)
        dice_original = dice_score(original_mask[:,:,0]//255,y_pred_original.astype(int))
        jaccard_original = jaccard_score(original_mask[:,:,0]//255,y_pred_original.astype(int), average="micro")
        
        validation_dice_original[i,cv_count] = dice_original
        validation_dice_resized[i,cv_count] = dice_resized
        validation_jaccard_original[i,cv_count] = jaccard_original
        validation_jaccard_resized[i,cv_count] = jaccard_resized
        
        if i < 5:
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            plt.imshow(original_img[...,::-1], 'gray', interpolation='none')
            plt.imshow(original_mask/255.0, 'jet', interpolation='none', alpha=0.4)
            plt.subplot(1,2,2)
            plt.imshow(original_img[...,::-1], 'gray', interpolation='none')
            plt.imshow(y_pred_original, 'jet', interpolation='none', alpha=0.4)
            plt.show()
            

    dice_resized_mean = validation_dice_resized[:,cv_count].mean()
    dice_original_mean = validation_dice_original[:,cv_count].mean()
    jaccard_resized_mean = validation_jaccard_resized[:,cv_count].mean()
    jaccard_original_mean = validation_jaccard_original[:,cv_count].mean()
        
    print("--------------------------------------")
    print("Mean validation DICE (on resized data):", dice_resized_mean) 
    print("Mean validation DICE (on original data):", dice_original_mean)
    print("--------------------------------------")
    print("Mean validation Jaccard (on resized data):", jaccard_resized_mean) 
    print("Mean validation Jaccard (on original data):", jaccard_original_mean)
    print("--------------------------------------")
    runtime = time.time() - time_start 
    print('Runtime: {} sec'.format(runtime))
    cv_count +=1
    all_history.append(history.history["val_dice_coef"])