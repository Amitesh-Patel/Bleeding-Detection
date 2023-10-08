import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.framework()

my_model = "efficientnetb1"
model = sm.Unet(my_model, encoder_weights="imagenet", input_shape=( 256,256, 3), classes=3, activation='sigmoid')
model.load_weights("Submission_segmentation/weights/unet_model_weights.h5")

#test for single image
def visualize_single_image_prediction(model, image_path):
    # Load the image
    original_img = cv2.imread(image_path)
    resized_img = cv2.resize(original_img, (256, 256))

    # Predict the mask
    X = np.expand_dims(resized_img, 0)
    y_pred = model.predict(X)
    _, y_pred_thr = cv2.threshold(y_pred[0, :, :, 0] * 255, 127, 255, cv2.THRESH_BINARY)
    y_pred = (y_pred_thr / 255).astype(int)

    # Resize the predicted mask back to the original size
    y_pred_original = cv2.resize(y_pred.astype(float), (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Visualize the original image and predicted mask
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(132)
    plt.title("Predicted Mask")
    plt.imshow(y_pred_original, cmap="gray")
    plt.axis("off")

    plt.show()

#example
visualize_single_image_prediction(model, "img- (10).png")
print("done")