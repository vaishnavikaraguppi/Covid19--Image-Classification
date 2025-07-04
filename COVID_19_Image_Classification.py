#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Image Classification

# ## Problem Statement

# ### Context

# Covid-19 is a fast-growing disease that affects human health severly. Patients diagonised with this condition suffers from lung infection. The medical community has recently released vaccines which have a slower effect in increasing the immunity. This virus has impacted various countries' human health and financial standing.  <br>
# 
# Deep learning algorithms have recently used image classification to identify medical images. Convolutional Neural Networks (CNN) can be widely utilized to identify COVID-19 to assist radiologists in medical analysis by classifying patients who are healthy, have viral pneumonia, or are affected by COVID using X-ray pictures of the lungs.

# ### Objective

# The aim of this project is to **Build a Convolutional Neural Network to differentiate an X-ray image of a person affected with covid from that of a healthy person or a person who has viral pneumonia(fever).**

# ### Data Dictionary

# - This dataset contains training set images of 3 classes which are converted into numpy arrays.
# 
# - The dataset comprises 3 classes:
#   - COVID-19: The patient who is effected due to covid.
#   - Viral Pneumonia: This is a viral fever which has similar characteristics like fever and cought that of Covid but is not covid.
#   - Normal- A healthy Person with no symptoms of covid or fever.
# 
# - The data file names are:
#   - CovidImages.npy
#   - CovidLabels.csv

# ####**Note: Please use GPU runtime to execute the code efficiently**

# ## Importing necessary libraries

# In[ ]:


# Installing the libraries with the specified version.
#!pip install tensorflow==2.15.0 scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==1.5.3 opencv-python==4.8.0.76 -q --user


# In[ ]:


# Installing the libraries with the specified version.
#!pip install tensorflow==2.13.0 scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==1.5.3 opencv-python==4.8.0.76 -q --user


# **Note**: *After running the above cell, kindly restart the notebook kernel and run all cells sequentially from the start again.*

# In[1]:


import os
import numpy as np                                                                               # Importing numpy for Matrix Operations
import pandas as pd                                                                              # Importing pandas to read CSV files
import matplotlib.pyplot as plt                                                                  # Importing matplotlib for Plotting and visualizing images
import math                                                                                      # Importing math module to perform mathematical operations
import cv2                                                                                       # Importing openCV for image processing
import seaborn as sns                                                                            # Importing seaborn to plot graphs


# Tensorflow modules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator                              # Importing the ImageDataGenerator for data augmentation
from tensorflow.keras.models import Sequential                                                   # Importing the sequential module to define a sequential model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization # Defining all the layers to build our CNN Model
from tensorflow.keras.optimizers import Adam,SGD                                                 # Importing the optimizers which can be used in our model
from sklearn import preprocessing                                                                # Importing the preprocessing module to preprocess the data
from sklearn.model_selection import train_test_split                                             # Importing train_test_split function to split the data into train and test
from sklearn.metrics import confusion_matrix                                                     # Importing confusion_matrix to plot the confusion matrix
from sklearn.preprocessing import LabelBinarizer
# Display images using OpenCV
#from google.colab.patches import cv2_imshow                                                      # Importing cv2_imshow from google.patches to display images
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from keras.callbacks import ReduceLROnPlateau
import random
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[ ]:


# Mount Google drive to access the dataset
#from google.colab import drive
#drive.mount('/content/drive')


# In[3]:


# Load the image file of the dataset
images = np.load(r"C:\Users\karag\Downloads\Covid 19 Project\CovidImages.npy")
# Load the labels file of the dataset
labels = pd.read_csv(r"C:\Users\karag\Downloads\Covid 19 Project\CovidLabels.csv")


# ## Data Overview

# ### Understand the shape of the dataset

# In[4]:


print(images.shape)
print(labels.shape)         


# ## Exploratory Data Analysis

# ### Plotting random images from each of the class

# In[5]:


def plot_images(images,labels):
  num_classes=10                                                                  # Number of Classes
  categories=np.unique(labels)
  keys=dict(labels['Label'])                                                      # Obtaing the unique classes from y_train
  rows = 3                                                                        # Defining number of rows=3
  cols = 4                                                                        # Defining number of columns=4
  fig = plt.figure(figsize=(10, 8))                                               # Defining the figure size to 10x8
  for i in range(cols):
      for j in range(rows):
          random_index = np.random.randint(0, len(labels))                        # Generating random indices from the data and plotting the images
          ax = fig.add_subplot(rows, cols, i * rows + j + 1)                      # Adding subplots with 3 rows and 4 columns
          ax.imshow(images[random_index, :])                                      # Plotting the image
          ax.set_title(keys[random_index])
  plt.show()


# In[6]:


plot_images(images, labels) 


# ### Checking the distribution of the target variable

# In[7]:


sns.countplot(x=labels['Label'])        
plt.xticks(rotation='vertical')


# ## Data Pre-Processing

# ### Converting the BGR images to RGB images.

# In[8]:


# Converting the images from BGR to RGB using cvtColor function of OpenCV
for i in range(len(images)):
  images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)        


# ### Resizing images

# As the size of the images is large, it may be computationally expensive to train on these larger images; therefore, it is preferable to reduce the image size from 128 to 64.

# In[9]:


images_decreased=[]
height = 64                    
width =  64                   
dimensions = (width, height)
for i in range(len(images)):
  images_decreased.append( cv2.resize(images[i], dimensions, interpolation=cv2.INTER_LINEAR))


# **Image before resizing**

# In[10]:


plt.imshow(images[3])


# **Image after resizing**

# In[11]:


plt.imshow(images_decreased[3])


# ### Data Preparation for Modeling

# - As we have less images in our dataset, we will only use 10% of our data for testing, 10% of our data for validation and 80% of our data for training.
# - We are using the train_test_split() function from scikit-learn. Here, we split the dataset into three parts, train,test and validation.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(np.array(images_decreased),labels['Label'] , test_size=0.1, random_state=42,stratify=labels)


# ### Encoding the target labels

# In[13]:


# Convert labels from names to one hot vectors.
# We have already used encoding methods like onehotencoder and labelencoder earlier so now we will be using a new encoding method called labelBinarizer.
# Labelbinarizer works similar to onehotencoder

enc = LabelBinarizer()                                      
y_train_encoded = enc.fit_transform(y_train)         
y_test_encoded=enc.transform(y_test)              


# In[14]:


y_train_encoded.shape
y_test.shape


# ### Data Normalization

# Since the **image pixel values range from 0-255**, our method of normalization here will be **scaling** - we shall **divide all the pixel values by 255 to standardize the images to have values between 0-1.**

# In[15]:


# Complete the code to normalize the image pixels of train, and test
train_normalized = X_train.astype('float32')/255.0
X_test_normalized = X_test.astype('float32')/255.0


# ## Model Building

# In[16]:


# Clearing backend
backend.clear_session()


# In[17]:


# Fixing the seed for random number generators
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# In[18]:


# Intializing a sequential model
model1 = Sequential()                       

# Input_shape denotes input image dimension of images
model1.add(Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=(64, 64, 3)))

# add the max pooling to reduce the size of output of first conv layer
model1.add(MaxPooling2D((2, 2), padding = 'same'))

# create two similar convolution and max-pooling layers activation = relu
model1.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model1.add(MaxPooling2D((2, 2), padding = 'same'))

model1.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model1.add(MaxPooling2D((2, 2), padding = 'same'))

# flatten the output of the conv layer after max pooling to make it ready for creating dense connections
model1.add(Flatten())

# add a fully connected dense layer with 16 neurons
model1.add(Dense(16, activation='relu'))
model1.add(Dropout(0.3))
# add the output layer with 3 neurons and activation functions as softmax since this is a multi-class classification problem
model1.add(Dense(3, activation='softmax'))

# use the Adam Optimizer
opt=Adam()
# Compile the model using suitable metric for loss fucntion
model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()


# <b> Fitting the model on the train data

# In[19]:


history_1 = model1.fit(
            train_normalized, y_train_encoded,
            epochs=30,
            validation_split=0.1,
            shuffle=False,
            batch_size=64,
            verbose=1
)


# **Model Evaluation**

# In[20]:


plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# **Evaluate the model on test data**

# In[21]:


accuracy = model1.evaluate(X_test_normalized, y_test_encoded, verbose=2) 


# **Plotting the Confusion Matrix**

# In[22]:


# Here we would get the output as probablities for each category
y_pred=model1.predict(X_test_normalized)


# In[23]:


# Obtaining the categorical values from y_test_encoded and y_pred
y_pred_arg=np.argmax(y_pred,axis=1)
y_test_arg=np.argmax(y_test_encoded,axis=1)

# Plotting the Confusion Matrix using confusion matrix() function which is also predefined in tensorflow module
confusion_matrix = tf.math.confusion_matrix(y_test_arg, y_pred_arg)
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.4,
    fmt="d",
    square=True,
    ax=ax
)
# Setting the labels to both the axes
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Covid', 'Normal', 'Viral Pneumonia'],rotation=20)
ax.yaxis.set_ticklabels(['Covid', 'Normal', 'Viral Pneumonia'],rotation=20)
plt.show()


# **Plotting Classification Report**

# In[24]:


# Plotting the classification report
from sklearn import metrics
cr=metrics.classification_report(y_test_arg, y_pred_arg)    
print(cr)


# ## Model Performance Improvement

# **Reducing the Learning Rate:**
# 
# **ReduceLRonPlateau()** is a function that will be used to decrease the learning rate by some factor, if the loss is not decreasing for some time. This may start decreasing the loss at a smaller learning rate. There is a possibility that the loss may still not decrease. This may lead to executing the learning rate reduction again in an attempt to achieve a lower loss.

# In[25]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)


# ### **Data Augmentation**

# In[26]:


# Clearing backend
from tensorflow.keras import backend
backend.clear_session()

# Fixing the seed for random number generators
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# In[27]:


# All images to be rescaled by 1/255.
train_datagen = ImageDataGenerator(
                              rotation_range=20,
                              fill_mode='nearest'
                              )
# test_datagen  = ImageDataGenerator(rescale = 1.0/255.)


# In[28]:


# Intializing a sequential model
model2 = Sequential()

# add the first conv layer with 64 filters and kernel size 3x3 , padding 'same' provides the output size same as the input size
# Input_shape denotes input image dimension images
model2.add(Conv2D(64, (3,3), activation='relu', padding="same", input_shape=(64, 64, 3)))

# add max pooling to reduce the size of output of first conv layer
model2.add(MaxPooling2D((2, 2), padding = 'same'))


model2.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model2.add(MaxPooling2D((2, 2), padding = 'same'))
model2.add(BatchNormalization())

# flattening the output of the conv layer after max pooling to make it ready for creating dense connections
model2.add(Flatten())

# Adding a fully connected dense layer with 16 neurons
model2.add(Dense(16, activation='relu'))

# add dropout with dropout_rate=0.3
model2.add(Dropout(0.3))
# add the output layer with 3 neurons and activation functions as softmax since this is a multi-class classification problem
model2.add(Dense(3, activation='softmax'))

# initialize Adam Optimimzer
opt=Adam()
# Compile model
model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Generating the summary of the model
model2.summary()


# In[29]:


# Epochs
epochs = 30
# Batch size
batch_size = 64


history = model2.fit(train_datagen.flow(train_normalized,y_train_encoded,
                                       batch_size=batch_size,
                                       shuffle=False),
                                       epochs=epochs,
                                       steps_per_epoch= train_normalized.shape[0] // batch_size,
                                       validation_data=(X_test_normalized,y_test_encoded),
                                       verbose=1,callbacks=[learning_rate_reduction])


# **Model Evaluation**

# In[30]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# **Evaluate the model on test data**

# In[31]:


accuracy = model2.evaluate(X_test_normalized, y_test_encoded, verbose=2) 


# **Plotting the Confusion Matrix**

# In[32]:


y_pred=model2.predict(X_test_normalized)


# In[33]:


# Obtaining the categorical values from y_test_encoded and y_pred
y_pred_arg=np.argmax(y_pred,axis=1)
y_test_arg=np.argmax(y_test_encoded,axis=1)

# Plotting the Confusion Matrix using confusion matrix() function which is also predefined in tensorflow module
confusion_matrix = tf.math.confusion_matrix(y_test_arg,y_pred_arg)    
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.4,
    fmt="d",
    square=True,
    ax=ax
)
# Setting the labels to both the axes
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Covid', 'Normal', 'Viral Pneumonia'],rotation=20)
ax.yaxis.set_ticklabels(['Covid', 'Normal', 'Viral Pneumonia'],rotation=20)
plt.show()


# **Plotting Classification Report**

# In[34]:


cr=metrics.classification_report(y_test_arg, y_pred_arg)    # Complete the code to plot the classification report
print(cr)


# ## Final Model

# Comment on the final model you have selected and use the same in the below code to visualize the image.

# ### Visualizing the prediction

# In[35]:


# Visualizing the predicted and correct label of images from test data
plt.figure(figsize=(2,2))
plt.imshow(X_test[2])
plt.show()

print('Predicted Label', enc.inverse_transform(model1.predict((X_test_normalized[2].reshape(1,64,64,3)))))
print('True Label', enc.inverse_transform(y_test_encoded)[2])

plt.figure(figsize=(2,2))
plt.imshow(X_test[10])
plt.show()
## predict the test data using the final model selected
print('Predicted Label', enc.inverse_transform(model1.predict((X_test_normalized[10].reshape(1,64,64,3)))))  # Changed from 33 to 10
print('True Label', enc.inverse_transform(y_test_encoded)[10])

plt.figure(figsize=(2,2))
plt.imshow(X_test[20])
plt.show()
## predict the test data using the final model selected
print('Predicted Label', enc.inverse_transform(model1.predict((X_test_normalized[20].reshape(1,64,64,3)))))  # Changed from 59 to 20
print('True Label', enc.inverse_transform(y_test_encoded)[20])

plt.figure(figsize=(2,2))
plt.imshow(X_test[5])
plt.show()
## predict the test data using the final model selected
print('Predicted Label', enc.inverse_transform(model1.predict((X_test_normalized[5].reshape(1,64,64,3)))))  # Changed from 36 to 5
print('True Label', enc.inverse_transform(y_test_encoded)[5])


# _____
