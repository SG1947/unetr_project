import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from patchify import patchify
from config import cf

def load_dataset(path, split=0.1):
    train_data_path = os.path.join(path, "ISBI2016_ISIC_Part1_Training_Data", "ISBI2016_ISIC_Part1_Training_Data")
    ground_truth_path = os.path.join(path, "ISBI2016_ISIC_Part1_Training_GroundTruth", "ISBI2016_ISIC_Part1_Training_GroundTruth")
    test_data_path = os.path.join(path, "ISBI2016_ISIC_Part1_Test_Data", "ISBI2016_ISIC_Part1_Test_Data")
    test_ground_truth_path = os.path.join(path, "ISBI2016_ISIC_Part1_Test_GroundTruth", "ISBI2016_ISIC_Part1_Test_GroundTruth")

    train_images = sorted(glob(os.path.join(train_data_path, "*.jpg")))
    train_masks = sorted(glob(os.path.join(ground_truth_path, "*_Segmentation.png")))

    test_images = sorted(glob(os.path.join(test_data_path, "*.jpg")))
    test_masks = sorted(glob(os.path.join(test_ground_truth_path, "*_Segmentation.png")))

    split_size = int(len(train_images) * split)
    train_x, valid_x = train_test_split(train_images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_masks, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_images, test_masks)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0

    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape(cf["flat_patches_shape"])
    y.set_shape([cf["image_size"], cf["image_size"], 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds
