from pathlib import Path

# Resolve dataset paths relative to this script so it runs locally.
PROJECT_ROOT = Path(__file__).resolve().parent
train_folder = PROJECT_ROOT / 'dataset' / 'train'
test_folder = PROJECT_ROOT / 'dataset' / 'test'
validate_folder = PROJECT_ROOT / 'dataset' / 'valid'

if not train_folder.exists() or not validate_folder.exists() or not test_folder.exists():
    raise FileNotFoundError(
        f"Expected dataset folders were not found under {PROJECT_ROOT / 'dataset'}"
    )

# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import utils

print("Libraries Imported")

# Set the image size for resizing
IMAGE_SIZE = (350, 350)

# Initialize the image data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the batch size for training
batch_size = 8

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    str(train_folder),
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

# Create the validation data generator
validation_generator = test_datagen.flow_from_directory(
    str(validate_folder),
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

# Set up callbacks for learning rate reduction, early stopping, and model checkpointing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.5, min_lr=0.000001)
early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(filepath='best_model.weights.h5', verbose=2, save_best_only=True, save_weights_only=True)

# Define the number of output classes
OUTPUT_SIZE = 4

# Load a pre-trained model (Xception) without the top layers and freeze its weights.
# If pretrained weights cannot be downloaded (e.g., SSL restrictions), fall back to random init.
try:
    pretrained_model = tf.keras.applications.Xception(
        weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
except Exception as exc:
    print(f"Could not download ImageNet weights ({exc}). Falling back to weights=None.")
    pretrained_model = tf.keras.applications.Xception(
        weights=None, include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
pretrained_model.trainable = False

# Create a new model with the pre-trained base and additional layers for classification
model = Sequential()
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_SIZE, activation='softmax'))

print("Pretrained model used:")
pretrained_model.summary()

print("Final model created:")
model.summary()

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the training and validation data generators
history = model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=50,
    callbacks=[learning_rate_reduction, early_stops, checkpointer],
    validation_data=validation_generator,
    validation_steps=20
)

print("Final training accuracy =", history.history['accuracy'][-1])
print("Final testing accuracy =", history.history['val_accuracy'][-1])

# Function to display training curves for loss and accuracy
def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

# Display training curves for loss and accuracy
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)

# Save the trained model
model.save(str(PROJECT_ROOT / 'trained_lung_cancer_model.h5'))

# Function to load and preprocess an image for prediction
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

class_labels = list(train_generator.class_indices.keys())

def predict_and_plot(img_path):
    img = load_and_preprocess_image(str(img_path), IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    print(f"{img_path.name} -> {predicted_label}")
    plt.imshow(image.load_img(str(img_path), target_size=IMAGE_SIZE))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Use one sample image per class from test data when running locally.
for class_dir in sorted(test_folder.iterdir()):
    if not class_dir.is_dir():
        continue
    sample_images = [p for p in class_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    if not sample_images:
        continue
    predict_and_plot(sample_images[0])
