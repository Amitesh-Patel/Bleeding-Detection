import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

sm.framework()

my_model = "efficientnetb1"
model = sm.Unet(my_model, encoder_weights="imagenet", input_shape=(256, 256, 3), classes=3, activation='sigmoid')
model.load_weights("Submission_segmentation/weights/unet_model_weights.h5")

# Define the directory to save the images
output_dir = "result/Test-Dataset2"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_single_image_prediction(model, image_path, output_dir):
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

    # Save the original image and predicted mask
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    # cv2.imwrite(output_image_path, cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(output_dir, f"predicted_mask_{os.path.basename(image_path)}"), y_pred_original * 255)


# for i in os.listdir("Submission_segmentation/data/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 2"):
#     model_path = "Submission_segmentation/data/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 2/"+i
#     save_single_image_prediction(model,model_path,output_dir)
#     print(i)
