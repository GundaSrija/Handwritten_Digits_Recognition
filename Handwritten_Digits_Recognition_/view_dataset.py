import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def show_sample_images(x_train, y_train, samples=25):
    plt.figure(figsize=(10, 10))
    for i in range(samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap='gray')
        plt.xlabel(f'Label: {y_train[i]}')
    plt.suptitle('Sample Images from MNIST Dataset')
    plt.show()

def show_digit_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.title('Distribution of Digits in Training Set')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()

def show_dataset_info(x_train, y_train, x_test, y_test):
    print("Dataset Information:")
    print(f"Training images shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test images shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"\nPixel value range: {x_train.min()} to {x_train.max()}")
    print(f"Number of classes: {len(np.unique(y_train))}")

if __name__ == "__main__":
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Show dataset information
    show_dataset_info(x_train, y_train, x_test, y_test)
    
    # Show sample images
    show_sample_images(x_train, y_train)
    
    # Show distribution of digits
    show_digit_distribution(y_train)

