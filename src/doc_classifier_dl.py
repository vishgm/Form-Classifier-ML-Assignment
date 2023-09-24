"""
Description: Code for Training and Predictions using Vgg16 Model - document classification
Author: Vishak G.
Last modified: 23-09-2023
"""

import os

# Deep learning libraries - Tensorflow
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D, Conv2D, Dropout, BatchNormalization
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
# from keras.utils import np_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import *

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Other libaries
from pdf2image import convert_from_path
import pytesseract
import tempfile
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Image augmentation
import cv2
import albumentations as A

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def show_file_counts(dir):
    
    path_dict = dict()
    
    # Loop through train directory to find number of pdfs for each class
    for path in os.listdir(dir):
        path_dict[path] = len(os.listdir(os.path.join(dir, path)))
    
    # Sort by descending values (Class with max number of pdf first)
    path_dict = dict(sorted(path_dict.items(), key=lambda x: x[1])[::-1])
    
    return path_dict

class VGG16DocumentClassifier:
    def __init__(self, train_path, test_path, is_train=None, saved_model_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.is_train = is_train
        self.saved_model_path = saved_model_path

        return

   
    def convert_pdf_to_images(self, train_pdfs_dir, output_dir=None):
       
       for form_category_name in os.listdir(train_pdfs_dir):

        # Image dir
        if output_dir:
            form_category_folder_name = os.path.join(output_dir, form_category_name)
        else:
            form_category_folder_name = os.path.join(self.train_path, form_category_name)

        # PDFs dir
        form_category_pdf_folder_name = os.path.join(train_pdfs_dir, form_category_name)
        
        print(form_category_folder_name, os.path.isdir(form_category_folder_name))

        if not os.path.isdir(form_category_folder_name):
            os.mkdir(form_category_folder_name)


        for pdf_path in tqdm(os.listdir(form_category_pdf_folder_name)):
            pdf_full_path = os.path.join(form_category_pdf_folder_name, pdf_path)      
            
            with tempfile.TemporaryDirectory() as path:
                images_from_path = convert_from_path(pdf_full_path, output_folder=form_category_folder_name, fmt='png', first_page=1, last_page=1)
                
 
        print("Number of PDFs present for each class (highest first): \n\n", show_file_counts(train_pdfs_dir))

        return
       
    def augment_images(self, source_path=None):

        if not source_path:
            source_path = self.train_path


        transform = A.Compose([
            A.RandomRotate90(),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1)
            ], p=0.2),
            A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.1)]),
            A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(clip_limit=2)]),
            A.HueSaturationValue(p=0.3)
        ])

        # Loop through train directory to find number of images for each class
        for idx, path in enumerate(os.listdir(source_path)):
            img_class_path = os.path.join(source_path, path)
            
            
            target_no_of_images = 199
            
            if path !=  'Form-13F':
                no_of_images = len(os.listdir(img_class_path))
                print(path)
                for img_path in tqdm(os.listdir(img_class_path)):
                    img_full_path = os.path.join(img_class_path, img_path)
                    # for i in range(target_no_of_images - path_dict[path] + 1):
                    batch_size = round((target_no_of_images - no_of_images) /no_of_images)
                    for i in range(0, batch_size+1):
                        image = cv2.imread(img_full_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # print(image)
                        augmented_image = transform(image=image)['image']
                        cv2.imwrite(os.path.join(img_class_path, f"augmented_{i}"+ img_path), augmented_image)       


        
        print(show_file_counts(source_path))

        return
    
    def train(self, batch_size, no_of_classes):

        # Create Train, test ImageGenerator
        train_datagen = ImageDataGenerator()
        

        # Load images into DataLoader
        train_loader = train_datagen.flow_from_directory(self.train_path, batch_size=batch_size)
 

        # Initialize VGG16 Model
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256,3)) 

        # Add Dense Layers for finetuning
        x = vgg16_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(no_of_classes, activation='softmax')(x)

        model_final = Model(vgg16_model.input, predictions)

        # Modifying first 15 layers to be non-trainable
        for layer in model_final.layers[:15]:
            layer.trainable = False

        for idx, layer in enumerate(model_final.layers):
            print(idx, layer.name, layer.trainable)

        # Display final model summary
        model_final.summary()

        lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)

        # Set Learning rate for the model
        learning_rate = 5e-5
        model_final.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

        
        print(tf.config.list_physical_devices())


        history = model_final.fit(train_loader, epochs=5,shuffle=True, callbacks=[lr_reduce])

        # Plot Accuracy aand Loss curves for trained model
        acc = history.history['accuracy']
        loss = history.history['loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc)
        plt.title('Training Accuracy')
        plt.figure()

        plt.plot(epochs, loss)
        plt.title('Training loss')

        return
    

    def predict_test_data(self, model=None):
    
        test_datagen = ImageDataGenerator()
        test_loader = test_datagen.flow_from_directory(self.test_path, shuffle=False)
        batch_size = 32


        model = tf.keras.models.load_model(self.saved_model_path)

        # Predictions on test data
        y_true = test_loader.classes
        y_pred_probs = model.predict(test_loader, steps=test_loader.n // batch_size + 1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=test_loader.class_indices.keys()))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        # predictions = [labels[k] for k in predicted_class_indices]
        # print(predictions[:10])



        return
    
    def predict(self, img_path, model=None, classes=None):
        print(self.saved_model_path)

        labels = classes

        if self.saved_model_path:
            model = tf.keras.models.load_model(self.saved_model_path)

        
        img = image.load_img(img_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        ind_pred = model.predict(x)

        y_pos = np.arange(len(labels))
        score = ind_pred
        score_up = np.ravel(score)
        # plt.barh(y_pos, score_up, align='center', alpha=0.5)
        # plt.yticks(y_pos, labels)
        # plt.title("Prediction results")
        # plt.show()
        # plt.imshow(img)
        # plt.show()
        print("predicted", score)
        
        # predicted_class_indices = np.argmax(ind_pred, axis=1)
        # print(predicted_class_indices)
        # pred_label = [labels[k] for k in predicted_class_indices]

        pred_dict = {labels[i]: float(val) for i, val in enumerate(ind_pred[0])}

        print(pred_dict)

        return pred_dict
    
if __name__ == "__main__":
    img_path = r"E:\Github\ML-NLP-Assignment\data\a4911eeb-7eb5-479a-a29c-1dd0f61e2d72-1.png"
    labels = ['82-Submission-sheet', 'Form-13F', 'Form-19B', 'Form-D', 'Form-TA', 'Form-X', 'Others']

    vgg_model = VGG16DocumentClassifier(train_path="../data/train", test_path="../data/test", is_train=False, saved_model_path="../saved_models/Doc_classifier_vgg16")
    vgg_model.predict_test_data()