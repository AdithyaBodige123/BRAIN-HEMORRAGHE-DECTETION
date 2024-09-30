# main.py
import sys
import os
import keyboard  # Import the keyboard library

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from Code1.Cnn import cnnModel
from Code1.Classifiers.Adaboost import adaBoost
from Code1.Classifiers.Random_Forest import random_forest
from Code1.Classifiers.Svm import svmlinear, svmpoly, svmrbf, svmsigmoid
from Code1.Classifiers.DecisionTreeClassifier import decisionTreeClassifier
from Code1.Classifiers.Knn import knn, knnEMD
import Code1.Extract as ex
import numpy as np
import glob
import cv2
import pandas as pd

"""
Shuffle the data and split it into train & test sets.
Parameters:
    X: numpy matrix, representing the images as vectors - each row is the image features.
    Y: numpy vector of the labels.
Returns:
    trainX
    trainY
    testX
    testY 
    imagesTest - images of test set at original size - for the Draw method.
"""
def splitTestTrain(X, Y):
    trainSize = int(0.8 * X.shape[0])
    Y = np.reshape(Y, (Y.shape[0], 1))
    
    # Dynamic indexing based on dataset size
    indexes = np.arange(X.shape[0])
    indexes = np.reshape(indexes, (X.shape[0], 1))
    
    # Concatenate features with labels and indexes
    data = np.concatenate((X, Y, indexes), axis=1)
    
    # Shuffle the data
    np.random.shuffle(data)
    
    # Split into training and testing sets
    trainX = data[:trainSize, :-2]
    trainY = data[:trainSize, -2]
    testX = data[trainSize:, :-2]
    testY = data[trainSize:, -2]
    imagesTest = data[trainSize:, -1].astype(int)  # Ensure integer indices
    
    return trainX, trainY, testX, testY, imagesTest

"""
Main experiment: 
"""
if __name__ == '__main__':
    
    # 1) Load images and labels
    pathX = "C:\\Users\\DELL\\Desktop\\Head-CT-hemorrhage-detection-master\\Head-CT-hemorrhage-detection-master\\Dataset\\head_ct\\*.png"
    pathY = "C:\\Users\\DELL\\Desktop\\Head-CT-hemorrhage-detection-master\\Head-CT-hemorrhage-detection-master\\Dataset\\labels.csv"

    files = sorted(glob.glob(pathX))
    
    # Initialize list to hold processed images
    images = []
    
    # Define target size for resizing images
    TARGET_SIZE = (256, 256)  # (width, height)
    
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to load image at {path}")
            sys.exit(1)  # Exit if any image fails to load
        # Resize image to TARGET_SIZE
        img_resized = cv2.resize(img, TARGET_SIZE)
        images.append(img_resized)
    
    # Convert list of images to a NumPy array
    images = np.array(images)
    
    # Convert labels
    labels_df = pd.read_csv(pathY)
    
    # Remove leading/trailing spaces from column names
    labels_df.columns = labels_df.columns.str.strip()
    
    # Reference the correct column name
    labels = np.array(labels_df['hemorrhage'].tolist())
    
    # Verify that the number of labels matches the number of images
    if len(labels_df) != len(files):
        print(f"Error: Number of labels ({len(labels_df)}) does not match number of images ({len(files)}).")
        sys.exit(1)
    
    # Verify that all images have the same shape
    if len(set(img.shape for img in images)) != 1:
        print("Error: Not all images have the same dimensions after resizing.")
        sys.exit(1)
    
    print("All images successfully loaded and resized.")
    print()
    
    # Run on variety of image size options:
    for s in range(20, 150, 10):
        # 2) Extract the features with one of four methods: 'SIMPLE', 'HISTOGRAM', 'HUMOMENTS' and 'PAC'.
        # see 'Extract' doc.
        method_to_extract_features = ex.Method.HUMOMENTS
        X = ex.extract_features(images, method=method_to_extract_features, size=(s, s))
        print('Extract features method:', method_to_extract_features, ", image size:", s)

        # 3) Split data into train & test sets, including shuffle of the data
        trainX, trainY, testX, testY, testIm = splitTestTrain(X, labels)

        # 4) Train the models
        print('Begins testing the models...')

        results = np.zeros(9)
        nb_iteration = 10

        for epoch in range(nb_iteration):
            # Check for 'n' key press to move to the next image
            if keyboard.is_pressed('n'):  # Move to the next image
                print("Moving to the next image.")
                break  # Exit the loop for the current epoch
            
            # Check for 'q' key press to stop the process
            if keyboard.is_pressed('q'):  # If 'q' is pressed, exit the program
                print("Process stopped by user.")
                sys.exit(0)
            
            # Train the models as normal
            results[0] += knnEMD(trainX, trainY, testX, testY, images, testIm, numNeigh=2)
            results[1] += knn(trainX, trainY, testX, testY, images, testIm, numNeigh=2)
            results[2] += svmlinear(trainX, trainY, testX, testY, images, testIm)
            results[3] += svmpoly(trainX, trainY, testX, testY, images, testIm)
            results[4] += svmrbf(trainX, trainY, testX, testY, images, testIm)
            results[5] += svmsigmoid(trainX, trainY, testX, testY, images, testIm)
            results[6] += random_forest(trainX, trainY, testX, testY, images, testIm)
            results[7] += decisionTreeClassifier(trainX, trainY, testX, testY, images, testIm)
            results[8] += adaBoost(trainX, trainY, testX, testY, images, testIm)

        results = np.divide(results, nb_iteration)
        print('=============================')        
        print('The average results for standard machine learning models:')
        print('knn-EMD: {:.2f}%'.format(results[0]))
        print('knn: {:.2f}%'.format(results[1]))
        print('svm-linear: {:.2f}%'.format(results[2]))
        print('svm-poly: {:.2f}%'.format(results[3]))
        print('svm-RBF: {:.2f}%'.format(results[4]))
        print('svm-sigmoid: {:.2f}%'.format(results[5]))
        print('random forest: {:.2f}%'.format(results[6]))
        print('decision tree: {:.2f}%'.format(results[7]))
        print('adaBoost: {:.2f}%'.format(results[8]))
        print('=============================')        
        print()

    # Train the CNN model after evaluating traditional classifiers
    cnnModel(s, s, pathX, pathY)
