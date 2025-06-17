import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import CSVLogger

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

# Basic CNN classification model referenced from https://www.tensorflow.org/tutorials/images/classification
# Keras image parameters
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

# CSV logger code referenced from https://stackoverflow.com/questions/47843265/how-can-i-get-a-keras-models-history-after-loading-it-from-a-file-in-python?rq=3
# Create a CSV file to store the training history
csv_logger = CSVLogger('training.log', separator=',', append=False)

# Train the model
epochs = 15
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