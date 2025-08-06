#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:47:29 2022

@author: sophielanger
"""
# import relevant library
import numpy as np
import math
import statistics
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




def augment_and_resample(X_class, Y_class, n_samples, datagen):
    """
    Augment and resample the dataset to ensure required samples are fully visible.
    
    Args:
        X_class: Original image 
        Y_test: Corresponding label
        n_samples: number of augmented samples
        datagen: DataGenerator to deform images

    Returns:
        X_aug (numpy.ndarray): Array of augmented training images.
        Y_aug (numpy.ndarray): Array of corresponding class labels
    """
    X_aug, Y_aug = [], []
    max_attempts_per_sample = 20  # Maximum retries per sample
    attempts = 0

    while len(X_aug) < n_samples:
        # Ensure we are treating a single image as a batch
        original_images = np.expand_dims(X_class, axis=0)  # Adds an extra dimension
        original_labels = np.expand_dims(Y_class, axis=0)  # Adds an extra dimension

        # Generate augmented images in batches
        augmented_images = datagen.flow(
            np.expand_dims(original_images, axis=-1),
            original_labels,
            batch_size=len(original_images),
            shuffle=False
        )

        for batch_images, batch_labels in augmented_images:
            for i in range(len(batch_images)):
                if len(X_aug) >= n_samples:
                    break  # Stop once we have enough samples

                # Check if the image is fully visible
                image = batch_images[i, :, :, 0]
                if is_full_object_visible(image):
                    X_aug.append(image)
                    Y_aug.append(batch_labels[i])
                else:
                    attempts += 1
                    # Fallback to original image after max_attempts_per_sample
                    if attempts > max_attempts_per_sample:
                        if is_full_object_visible(original_images[i]):
                            X_aug.append(original_images[i])
                            Y_aug.append(original_labels[i])
                            attempts = 0  # Reset attempts for next sample

            if len(X_aug) >= n_samples:
                break  # Stop generating more images

    return np.array(X_aug), np.array(Y_aug)


def generate_data(n_train, n_test, num1, num2):
    """
    Generates the training and test data sets by augmenting images of two different classes from 
    MNIST.
    
    Args:
        n_train (int): Number of training samples per class
        n_test (int): Number of test samples per class
        num1 (int): class label of class 1 (digit between 0 to 9)
        num2 (int): class label of class 2 (digit between 0 and 0)
        
        pixels: A 2D binary array (e.g., image) where non-zero elements represent pixels of the object.
        
    Returns:
        X_train_aug (list of numpy.ndarray): Augmented training images.
        Y_train_aug (numpy.ndarray): Corresponding labels for training images.
        X_test_aug (list of numpy.ndarray): List of augmented test images.
        Y_test_aug (numpy.ndarray): Corresponding labels for test images.
    """
    
    # Load MNIST data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Filter data for the two specific classes (using only one image per class)
    filter1 = np.where(Y_train == num1)[0]
    filter2 = np.where(Y_train == num2)[0]


    # Choose the first image from each class for both training and testing
    X_class1, Y_class1 = X_train[filter1[0]], Y_train[filter1[0]]
    X_class2, Y_class2 = X_train[filter2[0]], Y_train[filter2[0]]

    # Augment data using the same image for both training and testing
    datagen_train = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[1.0, 2.0],
        brightness_range=[0.8, 1.5],
    )
    
    datagen_test = ImageDataGenerator(
        width_shift_range=0.2,  # Different augmentation for test set
        height_shift_range=0.2,
        zoom_range=[1.0, 1.5],
        brightness_range=[0.7, 1.2],
    )

    # Augment the same image for both training and testing
    X_train_aug1, Y_train_aug1 = augment_and_resample(X_class1, Y_class1, n_train, datagen_train)
    X_train_aug2, Y_train_aug2 = augment_and_resample(X_class2, Y_class2, n_train, datagen_train)
    
    X_test_aug1, Y_test_aug1 = augment_and_resample(X_class1, Y_class1, n_test, datagen_test)
    X_test_aug2, Y_test_aug2 = augment_and_resample(X_class2, Y_class2, n_test, datagen_test)

    # Combine data from both classes for training and testing
    X_train_aug = np.concatenate([X_train_aug1, X_train_aug2])
    Y_train_aug = np.concatenate([Y_train_aug1, Y_train_aug2])
    X_test_aug = np.concatenate([X_test_aug1, X_test_aug2])
    Y_test_aug = np.concatenate([Y_test_aug1, Y_test_aug2])

    return X_train_aug, Y_train_aug, X_test_aug, Y_test_aug

def is_full_object_visible(image):
    """
    Ensure that all non-zero pixels remain within the visible frame.
    
    Args:
        image (numpy.ndarray): 2D array representing the image.

    Returns:
        bool: True if the object is fully visible, False otherwise.
    """
    # Find the bounding box of non-zero pixels
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    visible_rows = np.where(rows)[0]
    visible_cols = np.where(cols)[0]

    if len(visible_rows) == 0 or len(visible_cols) == 0:
        return False  # Object not visible at all

    # Check if the bounding box touches the frame
    min_row, max_row = visible_rows[0], visible_rows[-1]
    min_col, max_col = visible_cols[0], visible_cols[-1]

    return (min_row > 0 and max_row < image.shape[0] and
            min_col > 0 and max_col < image.shape[1])


# Example usage:

X_train_aug, Y_train_aug, X_test_aug, Y_test_aug = generate_data(100, 20, 0, 1)

# Check generated sample sizes
print(f"Training samples generated: {len(X_train_aug)}")
print(f"Test samples generated: {len(X_test_aug)}")





def visualize_images(X_train, Y_train, X_test, Y_test, num1, num2, n_samples=5):
    """
    Visualizes the first few images from the training and test sets for both classes.
    
    Args:
        X_train_aug (numpy.ndarray): Training images.
        Y_train_aug (numpy.ndarray): Corresponding labels for training images.
        X_test_aug (numpy.ndarray): Test images. 
        Y_test_aug (numpy.ndarray): Corresponding labels for test images.
        num1 (int): Class label of class 1 (digit between 0 to 9)
        num2 (int): class label of class 2 (digit between 0 and 0)
        n_samples (int): number of augmented images to be plotted per class
        
    Returns:
        None: Displays plots of the images.
    """
    fig, ax = plt.subplots(2, n_samples, figsize=(12, 6))

    # Visualize the training images for both classes
    train_class1_images = X_train[Y_train == num1]
    train_class2_images = X_train[Y_train == num2]

    for i in range(min(n_samples, len(train_class1_images))):
        # Training images for class num1
        ax[0, i].imshow(train_class1_images[i].reshape(28, 28), cmap="gray")
        ax[0, i].set_title(f"Class {num1}")
        ax[0, i].axis('off')

    for i in range(min(n_samples, len(train_class2_images))):
        # Training images for class num2
        ax[1, i].imshow(train_class2_images[i].reshape(28, 28), cmap="gray")
        ax[1, i].set_title(f"Class {num2}")
        ax[1, i].axis('off')

    plt.show()

    # Visualize the test images for both classes
    test_class1_images = X_test[Y_test == num1]
    test_class2_images = X_test[Y_test == num2]

    fig, ax = plt.subplots(2, n_samples, figsize=(12, 6))

    for i in range(min(n_samples, len(test_class1_images))):
        # Test images for class num1
        ax[0, i].imshow(test_class1_images[i].reshape(28, 28), cmap="gray")
        ax[0, i].set_title(f"Class {num1}")
        ax[0, i].axis('off')

    for i in range(min(n_samples, len(test_class2_images))):
        # Test images for class num2
        ax[1, i].imshow(test_class2_images[i].reshape(28, 28), cmap="gray")
        ax[1, i].set_title(f"Class {num2}")
        ax[1, i].axis('off')

    plt.show()


# Visualize the first 4 training and test images for both classes
visualize_images(X_train_aug, Y_train_aug, X_test_aug, Y_test_aug, 0, 1, n_samples=4)


# Estimator 1: Image alignment
#compute the rectangular support
def rec_support(pixels):
    """
    Computes the rectangular bounding box of the non-zero elements in a 2D binary array.
    
    Args:
        pixels: A 2D binary array (e.g., image) where non-zero elements represent pixels of the object.
        
    Returns:
        list: [i_min, i_max, j_min, j_max] indicating the bounds of the rectangle:
              - i_min, i_max: Row indices of the top and bottom bounds.
              - j_min, j_max: Column indices of the left and right bounds.
    """
    # Compute row-wise bounds (min and max indices of non-zero elements)
    row_indices = np.any(pixels > 0, axis=1)
    i_min = np.argmax(row_indices)  # First row with non-zero element
    i_max = len(row_indices) - 1 - np.argmax(row_indices[::-1])  # Last row with non-zero element

    # Compute column-wise bounds (min and max indices of non-zero elements)
    col_indices = np.any(pixels > 0, axis=0)
    j_min = np.argmax(col_indices)  # First column with non-zero element
    j_max = len(col_indices) - 1 - np.argmax(col_indices[::-1])  # Last column with non-zero element

    return [i_min, i_max, j_min, j_max]


def Z(pixels):
    """
    Transforms the given pixel array into a normalized feature matrix Z.
    
    Args:
        pixels (numpy.ndarray): A 2D array of pixel intensities.
        
    Returns:
        numpy.ndarray or None: The normalized feature matrix Z, or None if the input is invalid.
    """
    size = pixels.shape[0]
    Z = np.zeros((size, size))
    i_min, i_max, j_min, j_max = rec_support(pixels)

    # Check for invalid support
    if i_min == 0 or i_max == size or j_min == 0 or j_max == size or j_min==0:
        return None

    # Generate Z matrix and collect non-zero pixel values
    values = []
    seq = [x/size for x in range(0, size)]

    for t1 in seq:
        for t2 in seq:
            i=math.ceil(i_min+t1*(i_max - i_min))
            j=math.ceil(j_min+t2*(j_max-j_min))
            n1 = int(t1 * size)
            n2 = int(t2 * size)
            Z[n1, n2] = pixels[i, j]
            values.append(pixels[i, j])
    # Normalize Z using the norm of pixel values
    norm_value = np.linalg.norm(Z)
    if norm_value > 0:
        Z /= norm_value

    return Z


# #missclassification loss
def loss_est1(X_train, Y_train, X, Y):
    """
    Estimates the classification loss using the image alignment approach.
    
    Args:
        X_train (numpy.ndarray): List of training images.
        Y_train (numpy.ndarray): Corresponding labels for training images.
        X (numpy.ndarray): List of test images.
        Y (numpy.ndarray): Corresponding labels for test images.
        
    Returns:
        float: The average misclassification loss (float).
    """
    n = len(X)
    loss = 0

    # Precompute Z-transform for training set
    transformed_train = []
    for x_train in X_train:
        z_train = Z(x_train)
        transformed_train.append(z_train if z_train is not None else None)

    for i, x in enumerate(X):
        z_x = Z(x)
        if z_x is None:
            continue

        # Find the closest training example in Z-space
        min_dist = float('inf')
        predicted_label = None

        for j, z_train in enumerate(transformed_train):
            if z_train is None:
                continue

            dist = np.linalg.norm(z_train - z_x)
            if dist < min_dist:
                min_dist = dist
                predicted_label = Y_train[j]

        # Increment loss if prediction is incorrect
        if predicted_label != Y[i]:
            loss += 1

    return loss / n




# CNN classifier

def CNN(X_train, Y_train, X_test, Y_test):
    """
    Trains a Convolutional Neural Network (CNN) with 1 convolutional layer with
    filter size 32 x 32, one max-pooling layer with patch size 2 x 2 and one
    fully connected layer with 128 neurons. Then evaluates its performance.
    
    Args:
        X_train (numpy.ndarray): Training images.
        Y_train (numpy.ndarray): Labels for the training images
        X_test (numpy.ndarray): Test images.
        Y_test (numpy.ndarray): Labels for the test images.
        
    Returns:
        tuple: Average misclassification loss (float) and a list of misclassified test images (list).
    """
    misclassified_images = []
    image_dim = X_train[0].shape[0]
    
    # Preprocessing: Reshape and normalize inputs
    X_train = np.reshape(X_train,(len(X_train), image_dim, image_dim, 1)) / 255.0
    X_test=np.reshape(X_test,(len(X_test), image_dim, image_dim, 1)) / 255.0
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # Model Architecture
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_dim, image_dim, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate misclassification loss
    loss = 0
    for i in range(len(X_test)):
        # Predict for each test image
        pred = model.predict(X_test[i].reshape(1, image_dim, image_dim, 1), verbose=0)
        predicted_label = np.argmax(pred)
        
        if predicted_label != Y_test[i]:
            misclassified_images.append(X_test[i])
            loss += 1

    # Normalize loss to calculate misclassification rate
    misclassification_loss = loss / len(X_test)
    
    return misclassification_loss, misclassified_images



def CNN1(X_train, Y_train, X_test, Y_test):
    """
    Trains a Convolutional Neural Network (CNN) with 3 convolutional layer and filter sizes
    32 x 32, 64 x 64 and 128 x 128 respectively, one max-pooling layer of patch size 2 x 2 and one
    fully connected layer with 128 neurons. Then evaluates its performance.

    Args:
        X_train (numpy.ndarray): Training images.
        Y_train (numpy.ndarray): Labels for the training images.
        X_test (numpy.ndarray): Test images.
        Y_test (numpy.ndarray): Labels for the test images.

    Returns:
        tuple: Misclassification loss (float) and a list of misclassified test images.
    """
    misclassified_images = []
    image_dim = X_train[0].shape[1]
    
    # Reshape and normalize the datasets
    X_train = np.reshape(X_train,(len(X_train), image_dim, image_dim, 1)) / 255.0
    X_test=np.reshape(X_test,(len(X_test), image_dim, image_dim, 1)) / 255.0
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_dim, image_dim, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the model on the test set
    misclassification_count = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i].reshape(1, image_dim, image_dim, 1), verbose=0)
        predicted_label = np.argmax(pred)
        
        if predicted_label != Y_test[i]:
            misclassified_images.append(X_test[i])
            misclassification_count += 1

    # Calculate misclassification loss
    misclassification_loss = misclassification_count / len(X_test)

    return misclassification_loss, misclassified_images


def CNN2(X_train, Y_train, X_test, Y_test):
    """
    Trains a Convolutional Neural Network (CNN) with 3 convolutional layer and filter sizes
    32 x 32, 64 x 64 and 128 x 128 respectively, one max-pooling layer of patch size 2 x 2 and two fully
    connected layer and two fully connected layers with 256 and 128 neurons respectively.
    Then evaluates its performance.

    Args:
        X_train (numpy.ndarray): Training images.
        Y_train (numpy.ndarray): Labels for the training images.
        X_test (numpy.ndarray): Test images.
        Y_test (numpy.ndarray): Labels for the test images.

    Returns:
        tuple: Misclassification loss (float) and a list of misclassified test images.
    """
    misclassified_images = []
    image_dim = X_train[0].shape[1]
    
    # Reshape and normalize the datasets
    X_train = np.reshape(X_train,(len(X_train), image_dim, image_dim, 1)) / 255.0

    X_test=np.reshape(X_test,(len(X_test), image_dim, image_dim, 1)) / 255.0
   
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_dim, image_dim, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Added Dropout for regularization
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the model on the test set
    misclassification_count = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i].reshape(1, image_dim, image_dim, 1), verbose=0)
        predicted_label = np.argmax(pred)
        
        if predicted_label != Y_test[i]:
            misclassified_images.append(X_test[i])
            misclassification_count += 1

    # Calculate misclassification loss
    misclassification_loss = misclassification_count / len(X_test)

    return misclassification_loss, misclassified_images


def loss_med(rep, n_train, n_test, num1, num2):
    """
    Computes the median and interquartile range (IQR) for loss values of the four estimators, 
    i.e., three different CNN structures and the image alignment classifier.

    Args:
        rep (int): Number of repetitions for generating data and computing losses.
        n_train (int): Number of training images.
        n_test (int): Number of test images.
        num1, num2 (int): The two different labels for the classes in the 
        classification problem.

    Returns:
        tuple: Medians and interquartile ranges for the losses of four estimators.
    """
    losses = {1: [], 2: [], 3: [], 4: []}

    # Compute losses for each repetition
    for _ in range(rep):
        X_train, Y_train, X_test, Y_test = generate_data(n_train, n_test, num1, num2)

        losses[1].append(loss_est1(X_train, Y_train, X_test, Y_test))
        losses[2].append(CNN(X_train, Y_train, X_test, Y_test)[0])
        losses[3].append(CNN1(X_train, Y_train, X_test, Y_test)[0])
        losses[4].append(CNN2(X_train, Y_train, X_test, Y_test)[0])

    # Calculate medians and interquartile ranges
    results = {}
    for key in losses:
        res_median = statistics.median(losses[key])
        q3, q1 = np.percentile(losses[key], [75, 25])
        iqr = q3 - q1
        results[key] = {"median": res_median, "iqr": iqr}

    # Extract results into returnable format
    res1, res2, res3, res4 = (results[i]["median"] for i in range(1, 5))
    iqr1, iqr2, iqr3, iqr4 = (results[i]["iqr"] for i in range(1, 5))

    return res1, res2, res3, res4, iqr1, iqr2, iqr3, iqr4



def summarize_diagnostics(rep, n_test, num1, num2):
    """
    Plots the misclassification error curves for the four different 
    estimators, i.e., the three different CNN structures and the image alignment
    approach and sample sizes [2, 4, 8, 16, 32, 64] for each class

    Args:
        rep (int): Number of repetitions for generating data and computing losses.
        n_test (int): Number of test samples.
        num1, num2 (int): Class labels for the two different classes in the classification
        problem.

    Returns:
        tuple: Histories of missclassification errors for each estimator.
    """
    # Initialize variables for storing results
    history_est1, history_cnn1, history_cnn2, history_cnn3 = [], [], [], []
    iqr_est1, iqr_cnn1, iqr_cnn2, iqr_cnn3 = [], [], [], []

    # Define sample sizes
    sample_sizes = [2, 4, 8, 16, 32, 64]
    half_sample_sizes = [s // 2 for s in sample_sizes]

    # Compute losses for each sample size
    for n_train in half_sample_sizes:
        res1, res2, res3, res4, iqr1, iqr2, iqr3, iqr4 = loss_med(rep, n_train, n_test, num1, num2)

        history_est1.append(res1)
        history_cnn1.append(res2)
        history_cnn2.append(res3)
        history_cnn3.append(res4)

        iqr_est1.append(iqr1)
        iqr_cnn1.append(iqr2)
        iqr_cnn2.append(iqr3)
        iqr_cnn3.append(iqr4)

    # Plotting the missclassification error
    fig, ax = plt.subplots()
    message = f"Missclassification Error for {num1} vs. {num2}"
    plt.title(message)
    
    # Plot the histories for each estimator
    ax.plot(sample_sizes, history_est1, label='IAC', marker='o')
    ax.plot(sample_sizes, history_cnn1, label='CNN1', marker='o')
    ax.plot(sample_sizes, history_cnn2, label='CNN2', marker='o')
    ax.plot(sample_sizes, history_cnn3, label='CNN3', marker='o')

    plt.xlabel('Training Sample Size')
    plt.ylabel('Missclassification Error')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.show()

    return history_est1, history_cnn1, history_cnn2, history_cnn3

        
print(summarize_diagnostics(10, 100, 0, 4))     