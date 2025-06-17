import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import seaborn as sns
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix,classification_report

# Import Dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

# Move data dir to the color folder
data_dir = pathlib.Path(path).with_suffix('')
data_dir_color = data_dir / 'plantvillage dataset' / 'color'

# Create directories to hold healthy and diseased
healthy_dir = data_dir_color / 'healthy'
healthy_dir.mkdir(exist_ok=True)

diseased_dir = data_dir_color / 'diseased'
diseased_dir.mkdir(exist_ok=True)

# If a directory is healthy it's moved to the healthy directory
for dir in data_dir_color.iterdir():
    if '_healthy' in dir.name:
        dir.rename(healthy_dir / dir.name)

# If a directory is diseased it's moved to the diseased directory
for dir in data_dir_color.iterdir():
    if '__' in dir.name:
        dir.rename(diseased_dir / dir.name)

# Keras image parameters
# Basic CNN image classification model referenced from https://www.tensorflow.org/tutorials/images/classification
batch_size = 32
img_height = 224
img_width = 224

# Training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_color,
    validation_split=0.2,
    subset="training",
    seed=21,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_color,
    validation_split=0.2,
    subset="validation",
    seed=21,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Store the class names
class_names = train_ds.class_names

# Normalize the data
normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Instantiate shuffling and prefetching of data
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Data augmentation to help overfitting
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Create the model
num_classes = len(class_names)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Create a CSV file to store the training history
# CSV logger code referenced from https://stackoverflow.com/questions/47843265/how-can-i-get-a-keras-models-history-after-loading-it-from-a-file-in-python?rq=3
csv_logger = CSVLogger('training.log', separator=',', append=False)

# # Create early stopping callback
# early_stopping = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

# Train the model
epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    validation_steps=len(val_ds) // batch_size,
    epochs=epochs,
    steps_per_epoch=len(train_ds) // batch_size,
    callbacks=[csv_logger]
)

# Save the model
model.save("models/crop_disease_detection_model.keras", overwrite=True)

# Plot model loss and accuracy
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

plt.savefig('figures/train_loss.png')
plt.show()

# Create arrays to store predicted and true labels for confusion matrix
# Classes extracted code referenced from https://stackoverflow.com/questions/64622210/how-to-extract-classes-from-prefetched-dataset-in-tensorflow-for-confusion-matri
y_pred = []
y_true = []

# Iterate over the validation dataset
for images, labels in val_ds:
    y_true.append(labels)
    # Use model to predict labels
    predictions = model.predict(images)
    y_pred.append(np.argmax(predictions, axis=-1))

# Convert the labels into tensors
correct_labels = tf.concat([item for item in y_true], axis=0)
predicted_labels = tf.concat([item for item in y_pred], axis=0)

# Create a confusion matrix
# Confusion Matrix referenced from https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
cm = confusion_matrix(correct_labels, predicted_labels)

# Plot the confusion matrix and use seaborn to create a heatmap
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
# Move the x-axis title and labels to the top
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()

plt.savefig('figures/confusion_matrix.png')
plt.show()

# Print a classification report
print(classification_report(correct_labels, predicted_labels))

# Create arrays to hold the disease names and number of images per disease
diseases = []
num_images = []

for child in diseased_dir.iterdir():
    if child.is_dir():
        diseases.append(child.name)
        num_images.append(len(os.listdir(child)))

# Create a horizontal bar chart for the number of disease images
# Bar chart code referenced from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
plt.barh(diseases, num_images)
plt.title('Number of Disease Images')
plt.ylabel('Diseases')
plt.xlabel('Images')

plt.savefig('figures/num_disease_images.png')
plt.show()