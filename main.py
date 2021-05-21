#importing a few required libraries

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

#creating a drive variable to access Google Drive

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#Downloading the dataset, using the ID of the file uploaded on Google Drive

download = drive.CreateFile({'id': ''})
download.GetContentFile('dataset')
download = drive.CreateFile({'id': ''})
download.GetContentFile('dataset')

#importing libraries for model building phase

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

train = pd.read_csv('train.csv')

# Reading training images saving into a list and converting it into a numpy array

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=False)  #greyscale set to false since we have coloured images
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

#Since multiclass variable one-hot encoding the target variable

y=train['label'].values
y = to_categorical(y)

#Creating a validation set from the training data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Defining the model structure using 2 convolutional layers, one dense hidden layer and an output layer

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Compiling the Model

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


#Training the Model

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#importing and downloading test cases

download = drive.CreateFile({'id': ''})
download.GetContentFile('')
test = pd.read_csv('') #importing test file

#reading and storing all test images

test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('test/'+test['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

# making predictions

prediction = model.predict_classes(test)

#downloading the solution file
download = drive.CreateFile({'id': '1z4QXy7WravpSj-S4Cs9Fk8ZNaX-qh5HF'})
download.GetContentFile('sample_submission_I5njJSF.csv')

sample = pd.read_csv('sample_submission_I5njJSF.csv')
sample['label'] = prediction
sample.to_csv('sample_cnn.csv', header=True, index=False)

#IWroteTheCodeButCouldntCompleteItFully.Itis95%CompletedJustImportingAndDownloadingFilesIsLeftOver.
 




