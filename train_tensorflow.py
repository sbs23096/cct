import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Paths to your image folders (train/test or just train for quick start)
train_dir = 'data/train'
val_dir = 'data/val'  # optional, if you have validation data

# Image parameters
img_size = (224, 224)
batch_size = 16

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # If no separate val_dir, split train data
    horizontal_flip=True,
    zoom_range=0.2 
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # binary classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load pretrained ResNet50 without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# Save model
model.save('cat_dog_model.h5')
