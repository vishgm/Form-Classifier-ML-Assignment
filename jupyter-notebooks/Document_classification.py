#!/usr/bin/env python
# coding: utf-8

# <h1><center> ML Assignment - Sept 2023
# </center></h1>

# ## Problem statement

# The objective of this assignment is to build an NLP solution for the provided dataset. 
# The dataset consists of scanned documents from an archive (https://www.sec.gov/Archives/edgar/vprr/index.html)

# - Build a predictive model that can categorize a form into one of multiple form types (such as Form D, Form X, Form 13F etc)

# #### Table of Contents

# * **Data Loading / Preparation:** Load the data and understand the basic structure of the dataset.
# 
# * **Data Preprocessing:** Since the data consist of only PDFs, we have to convert them into Images for further processing / model building steps
# 
# * **Feature Engineering (Image Augmentation / Transformation,if any):** Create new features that might help improve the model's performance. Also, select the most relevant features for the model.
# 
# * **Model Building:** Train the model using different machine learning algorithms.
# 
# * **Model Evaluation**: Evaluate the model's performance using appropriate metrics.
# 
# * **Model Tuning**: Tune the model's hyperparameters to improve its performance.
# 
# * **Model Deployment**: 

# ### Data Loading / Preparation

# #### Import libraries

# In[1]:


# Import libraries
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

# Other libaries
from pdf2image import convert_from_path
import pytesseract
import tempfile
from tqdm.notebook import tqdm

# Image augmentation
import cv2
import albumentations as A


# #### Set Train, Test paths

# In[2]:


root_dir = "../data"

train_pdfs_dir = os.path.join(root_dir, "train_pdfs")
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")


# In[3]:


### Note: All files were manually labelled into classes such as Form X, Form 13f, form D etc


# In[4]:


#### Check number of documents for each class (Category)


# In[8]:


def show_file_counts(dir):
    
    path_dict = dict()
    
    # Loop through train directory to find number of pdfs for each class
    for path in os.listdir(dir):
        path_dict[path] = len(os.listdir(os.path.join(dir, path)))
    
    # Sort by descending values (Class with max number of pdf first)
    path_dict = dict(sorted(path_dict.items(), key=lambda x: x[1])[::-1])
    
   

    return path_dict


# In[9]:


print("Number of PDFs present for each class (highest first): \n\n", show_file_counts(train_pdfs_dir))


# In[6]:


## Clearly Form-13F has the highest number of pdfs, 


# ### Convert PDFs to Image

# In[6]:


for form_category_name in os.listdir(train_pdfs_dir):

    # Image dir
    form_category_folder_name = os.path.join(train_dir, form_category_name)

    # PDFs dir
    form_category_pdf_folder_name = os.path.join(train_pdfs_dir, form_category_name)
    
    print(form_category_folder_name, os.path.isdir(form_category_folder_name))
    if not os.path.isdir(form_category_folder_name):
        os.mkdir(form_category_folder_name)


    for pdf_path in tqdm(os.listdir(form_category_pdf_folder_name)):
        pdf_full_path = os.path.join(form_category_pdf_folder_name, pdf_path)      
        
        with tempfile.TemporaryDirectory() as path:
            images_from_path = convert_from_path(pdf_full_path, output_folder=form_category_folder_name, fmt='png', first_page=1, last_page=1)
            


# In[10]:


# path_dict = dict()

# # Loop through train directory to find number of pdfs for each class
# for path in os.listdir(train_dir):
#     path_dict[path] = len(os.listdir(os.path.join(train_dir, path)))

# # Sort by descending values (Class with max number of pdf first)
# path_dict = dict(sorted(path_dict.items(), key=lambda x: x[1])[::-1])

print("Number of images present for each class (highest first): \n\n", show_file_counts(train_dir))


# In[11]:


## Data augmentation for classes with lesser data


# In[ ]:





# In[20]:


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

# Loop through train directory to find number of pdfs for each class
for idx, path in enumerate(os.listdir(train_dir)):
    img_class_path = os.path.join(train_dir, path)
    
    
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


# In[25]:


show_file_counts(train_dir)


# In[ ]:





# In[ ]:





# In[58]:


# Solution 1: NLP based - using Tesseract


# In[102]:


TESSERACT_PATH = r"E:\Github\ML-NLP-Assignment\Tesseract\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

from PIL import Image
import re 
from fuzzywuzzy import fuzz

form_classes = [x for x in show_file_counts(train_dir).keys()]
# print(form_classes)
if '82-Submission-sheet' in form_classes:
    form_classes.remove('82-Submission-sheet')
    form_classes.append('82-Submissions-facing-sheet')

# Loop through train directory to find number of pdfs for each class
for idx, path in enumerate([os.listdir(train_dir)[5]]):
    img_class_path = os.path.join(train_dir, path)
    for img_path in [os.listdir(img_class_path)[13]]:
        img_full_path = os.path.join(img_class_path, img_path)
        data = pytesseract.image_to_string(img_full_path,lang='eng', config='--psm 6')
        data_split = data.split("\n")
        data_cleaned = [re.sub(r'[^a-zA-Z0-9]'," ", x.lower()) for x in data_split] 
        # print(data)
        for idx, line in enumerate(data_cleaned):
            get_class_name = [class_name for class_name in form_classes if re.sub(r'[^a-zA-Z0-9]'," ", class_name.lower()) in line or fuzz.ratio(re.sub(r'[^a-zA-Z0-9]'," ", class_name.lower()), line) > 90 ]
            if get_class_name:
                print(get_class_name)
                break
            elif idx == len(data_cleaned) - 1 and not get_class_name:
                print("Others")
        temp = Image.open(img_full_path)
        temp.show()


# In[47]:


'82- SUBMISSIONS FACING SHEET'.lower()


# In[224]:


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[225]:


## Solution 2 - Using DL


# In[226]:


## Create train, test data loaders


# In[274]:


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_loader = train_datagen.flow_from_directory(train_dir, batch_size=16)
test_loader = test_datagen.flow_from_directory(test_dir)


# In[275]:


vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256,3)) 


# In[276]:


x = vgg16_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model_final = Model(vgg16_model.input, predictions)

for layer in model_final.layers[:15]:
    layer.trainable = False

for idx, layer in enumerate(model_final.layers):
    print(idx, layer.name, layer.trainable)


# In[277]:


900 / 2


# In[278]:


model_final.summary()


# In[279]:


lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)


# In[280]:


learning_rate = 5e-5
model_final.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])


# In[281]:


import tensorflow as tf
print(tf.config.list_physical_devices())


# In[282]:


history = model_final.fit(train_loader, epochs=5,shuffle=True, callbacks=[lr_reduce])


# In[283]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.title('Training Accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.title('Training loss')


# In[284]:


# history.history


# In[ ]:





# In[285]:


import numpy as np

# Predictions on test data
pred = model_final.predict(test_loader, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = list(train_loader.class_indices.keys())
print("labels:",labels)

predictions = [labels[k] for k in predicted_class_indices]
print(predictions[:10])


# In[ ]:





# In[297]:


model_final.save("Doc_classifier_final")


# In[286]:


import keras.utils as image


# In[293]:


# img_path = r"E:\Github\ML-NLP-Assignment\data\test\Form-13F\0b3b0f1e-1a18-43c6-833f-7ce283ac5862-1.png"
temp = convert_from_path(r"E:\Github\ML-NLP-Assignment\data\test_1.pdf", fmt='png', output_folder=r"E:\Github\ML-NLP-Assignment\data", first_page=1, last_page=1)


# In[294]:


img_path = r"E:\Github\ML-NLP-Assignment\data\a4911eeb-7eb5-479a-a29c-1dd0f61e2d72-1.png"


# In[295]:


img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(img, axis=0)

ind_pred = model_final.predict(x)

y_pos = np.arange(len(labels))
score = ind_pred
score_up = np.ravel(score)
plt.barh(y_pos, score_up, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.title("Prediction results")
plt.show()
plt.imshow(img)
plt.show()
print("predicted", score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




