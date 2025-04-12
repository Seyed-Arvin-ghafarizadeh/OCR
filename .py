import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Constants
DATADIR = "D:/Case Study/Project_Arvin/OCR/c"
CATEGORIES = ["1", "3", "5", "7", "9", "11", "13", "15", "16", "17", "19", "22", "23", "24", "25", "26",
              "28", "30", "32", "33", "34", "36", "38", "40", "42", "44", "46", "48", "50", "51", "54", "56",
              "58", "59", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70"]
IMG_SIZE = (64, 64)
MAX_IMAGES = 640  # Max number of images per category

# Initialize an empty list for training data
training_data = []


# Function to create training data
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to alphabet dir
        class_num = CATEGORIES.index(category)

        for index, img in enumerate(os.listdir(path)):
            if index >= MAX_IMAGES:  # Limit to 640 images per category
                break
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue

                # Resize and apply thresholding
                resized_img = cv2.resize(img_array, IMG_SIZE, interpolation=cv2.INTER_AREA)
                _, thresh_img = cv2.threshold(resized_img, 200, 255, cv2.THRESH_BINARY)

                training_data.append([thresh_img, class_num])  # Store image and its label
            except Exception as e:
                print(f"Error processing {img}: {e}")
                continue


# Create training data
create_training_data()

# Prepare data for training
data_array = [image[0] for image in training_data]
labels = [image[1] for image in training_data]

# Normalize data and reshape it for the model
data_array = np.asarray(data_array) / 255.0  # Normalize the images
data_array = data_array.reshape(data_array.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1)  # Reshape for CNN input
labels = np.asarray(labels)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data_array, labels, test_size=0.1, shuffle=True)

# Define the model
model = Sequential()
model.add(Conv2D(30, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to avoid overfitting and improve training efficiency
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_test, Y_test),
                    callbacks=[early_stopping])

# Plot training history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
