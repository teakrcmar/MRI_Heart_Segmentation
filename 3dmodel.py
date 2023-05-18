# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
import random
from tensorflow.keras.utils import to_categorical

image_dir = '/home/teakrcmar/Desktop/3D_Unet_HeartSegmentation/resized_images_x64'
label_dir = '/home/teakrcmar/Desktop/3D_Unet_HeartSegmentation/resized_labels_x64'

output_shape = (64,64,64)
class_num = 8

def get_image_data(image_paths):
    images = []
    for path in image_paths:
        image = nib.load(path)
        image_data = image.get_fdata()
        images.append(image_data)
    return images


def get_label_data(label_path):
    labels = []
    for path in label_path:
        label = nib.load(path)
        label_data = label.get_fdata()
        labels.append(label_data)
    return labels


def data_generator(image_paths, label_paths, batch_size):

    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_label_paths = label_paths[i:i+batch_size]

            batch_images = get_image_data(batch_image_paths)
            batch_labels = get_label_data(batch_label_paths)
                
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            
            yield batch_images, batch_labels

image_ids = [f.split('_')[2].split('.')[0] for f in os.listdir(image_dir)]
label_ids = [f.split('_')[2].split('.')[0] for f in os.listdir(label_dir)]

np.random.seed(123)
num_epochs = 30
batch_size = 1
val_split = 0.21

for epoch in range(num_epochs):
    np.random.shuffle(image_ids)

    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(image_ids, test_size=0.3, random_state=123)

    train_image_paths = [os.path.join(image_dir, f'resized_image_{image_id}.nii') for image_id in train_ids]
    train_label_paths = [os.path.join(label_dir, f'resized_label_{image_id}.nii') for image_id in train_ids]
    val_image_paths = [os.path.join(image_dir, f'resized_image_{image_id}.nii') for image_id in val_ids]
    val_label_paths = [os.path.join(label_dir, f'resized_label_{image_id}.nii') for image_id in val_ids]

    train_steps = len(train_image_paths) // batch_size
    val_steps = len(val_image_paths) // batch_size

    train_generator = data_generator(train_image_paths, train_label_paths, batch_size)
    val_generator = data_generator(val_image_paths, val_label_paths, batch_size)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Dropout, BatchNormalization, Cropping3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3DTranspose

def unet(input_size=(64, 64, 64, 1, 2), binary_model = False):
    inputs = Input((input_size[0], input_size[1], input_size[2], input_size[3]))

    # Downsample path
    conv1 = Conv3D(16, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(16, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(128, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    # Bottleneck
    conv5 = Conv3D(256, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(256, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.3)(conv5)

    #Decoder
    up6 = Conv3DTranspose(128, 2, strides=(2, 2, 2), padding='same')(drop5)
    merge6 = Concatenate(axis=4)([up6, drop4])
    conv6 = Conv3D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(128, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv3DTranspose(64, 2, strides=(2, 2, 2), activation='relu', padding='same')(conv6)
    merge7 = Concatenate(axis=4)([conv3, up7])
    conv7 = Conv3D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv3DTranspose(32, 2, strides=(2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(1, 1, 1))(conv7))
    merge8 = Concatenate(axis=4)([conv2, up8])
    conv8 = Conv3D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv3DTranspose(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate(axis=4)([conv1, up9])
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    if binary_model:
        outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv9)
    else:
      outputs = Conv3D(8, (1, 1, 1), activation="softmax")(conv9)

    model = Model(inputs, outputs)
    return model

input_shape = (64, 64, 64, 1)


model = unet(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=30,
    validation_data=val_generator,
    validation_steps=val_steps
)

import tensorflow.keras.backend as K

def dice_coeff(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def jaccard_coeff(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    jaccard = K.mean((intersection + smooth)/(union + smooth), axis=0)
    return jaccard

def evaluate(model, val_dataset):
    total_dice = 0
    total_jaccard = 0
    num_batches = 0
    for batch in val_dataset:
        x, y_true = batch
        x = tf.cast(x, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        y_pred = model.predict(x)
        y_pred = tf.cast(y_pred, tf.float64)
        dice = dice_coeff(y_true, y_pred)
        jaccard = jaccard_coeff(y_true, y_pred)
        total_dice += dice
        total_jaccard += jaccard
        num_batches += 1
    avg_dice = total_dice / num_batches
    avg_jaccard = total_jaccard / num_batches
    return {'dice_coefficient': avg_dice, 'jaccard_index': avg_jaccard}

precision = evaluate(model, val_data)
print(precision)

