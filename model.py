

import pickle
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Bidirectional, LSTM, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, ClippingDistortion, PolarityInversion, Gain, Reverse, TimeMask
from config import Config

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def load_data(file_path):
    """
    Loads the data from a specified file.

    Parameters:
    ----------
    file_path : str
        The path to the file containing the data.

    Returns:
    -------
    tuple
        A tuple containing the features (X) and labels (y).
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def augment_data(X_train, y_train):
    """
    Augments the training data using various audio transformations.

    Parameters:
    ----------
    X_train : np.ndarray
        The training features.
    y_train : np.ndarray
        The training labels.

    Returns:
    -------
    tuple
        A tuple containing the augmented training features and labels.
    """
    augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=0.5),
        PolarityInversion(p=0.5),
        Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
        Reverse(p=0.5),
        TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.5),
    ])
    
    X_train_augmented, y_train_augmented = [], []
    for x, y in zip(X_train, y_train):
        X_train_augmented.append(x)
        y_train_augmented.append(y)
        augmented_sample = augmentations(samples=x, sample_rate=Config.SAMPLE_RATE)
        X_train_augmented.append(augmented_sample)
        y_train_augmented.append(y)
    
    return np.array(X_train_augmented), np.array(y_train_augmented)

def build_model(input_shape, num_classes):
    """
    Builds and compiles a Convolutional Recurrent Neural Network model.

    Parameters:
    ----------
    input_shape : tuple
        The shape of the input data.
    num_classes : int
        The number of output classes.

    Returns:
    -------
    tensorflow.keras.Model
        The compiled model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.5, seed=random_seed),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))),
        Dropout(0.5, seed=random_seed),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def step_decay(epoch):
    """
    Learning rate schedule function that reduces the learning rate at specific intervals.

    Parameters:
    ----------
    epoch : int
        The current epoch number.

    Returns:
    -------
    float
        The updated learning rate.
    """
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    return initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))

def train_model(model, X_train, y_train, X_test, y_test, class_weights, epochs=100, batch_size=32):
    """
    Trains the model with the given data and parameters, controlling randomness for reproducibility.

    Parameters:
    ----------
    model : tensorflow.keras.Model
        The model to be trained.
    X_train : np.ndarray
        The training features.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The test features.
    y_test : np.ndarray
        The test labels.
    class_weights : dict
        The class weights to handle class imbalance.
    epochs : int, optional
        The number of epochs to train the model (default is 100).
    batch_size : int, optional
        The batch size for training (default is 32).

    Returns:
    -------
    tensorflow.keras.callbacks.History
        The training history of the model.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        LearningRateScheduler(step_decay, verbose=1)
    ]
    
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, 
                     batch_size=batch_size, callbacks=callbacks, class_weight=class_weights, 
                     shuffle=True)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model on training and test data.

    Parameters:
    ----------
    model : tensorflow.keras.Model
        The model to be evaluated.
    X_train : np.ndarray
        The training features.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The test features.
    y_test : np.ndarray
        The test labels.

    Returns:
    -------
    None
    """
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

def save_model(model, architecture_file, weights_file):
    """
    Saves the model architecture and weights to files.

    Parameters:
    ----------
    model : tensorflow.keras.Model
        The model to be saved.
    architecture_file : str
        The file to save the model architecture.
    weights_file : str
        The file to save the model weights.

    Returns:
    -------
    None
    """
    model.save(architecture_file)
    if not weights_file.endswith('.weights.h5'):
        raise ValueError(f"The weights filename must end in '.weights.h5'. Received: {weights_file}")
    model.save_weights(weights_file)

def main():
    """
    The main function to load data, augment data, build the model, train the model,
    evaluate the model, and save the model.

    Returns:
    -------
    None
    """
    # Load data
    X, y = load_data(Config.DATA_PATH)
    
    # Split data into training and test sets with a fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=random_seed, stratify=y)
    
    # Augment training data (optional)
    # X_train, y_train = augment_data(X_train, y_train)
    
    # Get input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y))
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(y), y=y)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Build and train the model
    model = build_model(input_shape, num_classes)
    train_model(model, X_train, y_train, X_test, y_test, class_weights)
    
    # Evaluate the model
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save the model architecture and weights
    save_model(model, Config.MODEL_PATH, Config.WEIGHTS_PATH)

if __name__ == "__main__":
    main()
