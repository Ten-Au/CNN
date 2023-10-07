# Code a convolutional neural network (CNN) and train it to recognize: keyboard, mouse and calculator.
# In both programs:
# - divide the image database in the following ratio: 20% validation, 80% training data;
# - plot training accuracy and validation accuracy vs epochs;
# - plot training loss and validation loss vs epochs;

# To import images as the database I propose command (but it is not obligatory):
# tf.keras.utils.image_dataset_from_directory()
# The application of the command you can find in the "02 Image classification.py" file in the "10 and 11 Convolutional Neural Networks" folder (our previous lab).

# CNN final validation accuracy should be above 0.8.
# CNN final validation loss should be below 0.5.
# CNN with over 250,000 trainable parameters OR more than 10 epochs are for maximum 8 points.
# CNN with below 250,000 trainable parameters AND no more than 10 epochs are for maximum 11 points.

# Program 2. (max 11 points) use photos in the "main_directory_10" folder as a database.
# Use data augmentation so that the initial database is at least 4 times larger.
# Validation loss and training loss should have similar values, which will prove that the CNN is not overfitting.

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Import the dataset
batch_size = 32
img_height = 64
img_width = 64
 

train_ds = tf.keras.utils.image_dataset_from_directory(
  "main_directory_10",
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

 

val_ds = tf.keras.utils.image_dataset_from_directory(
"main_directory_10",
  labels='inferred',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#%%

"""the class names in the class_names attribute on these datasets"""
class_names = train_ds.class_names
print(class_names)


"""the class names in the class_names attribute on these datasets"""
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(5):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
    
model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
])
    
# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()