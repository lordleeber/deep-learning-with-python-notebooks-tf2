

"""
# Getting started with neural networks: Classification and regression

## Classifying movie reviews: A binary classification example

### The IMDB dataset

**Loading the IMDB dataset**
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def build_model1():
    """### Building your model
        **Model definition**
        """
    model1 = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    """**Compiling the model**"""
    model1.compile(optimizer="rmsprop",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

    return model1


def build_model2():
    model2 = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model2.compile(optimizer="rmsprop",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
    return model2


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print(train_data[0])
    print(train_labels[0])

    zz = max([max(sequence) for sequence in train_data])
    print(zz)

    """**Decoding reviews back to text**"""
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

    """### Preparing the data
    **Encoding the integer sequences via multi-hot encoding**
    """
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    print(x_train[0])

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    """### Validating your approach
    **Setting aside a validation set**
    """
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    """**Training your model**"""
    model1 = build_model1()
    history = model1.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

    history_dict = history.history
    history_dict.keys()

    """**Plotting the training and validation loss**"""
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    """**Plotting the training and validation accuracy**"""
    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    """**Retraining a model from scratch**"""
    # model2 = build_model2()
    # model2.fit(x_train, y_train, epochs=4, batch_size=512)
    # results = model2.evaluate(x_test, y_test)
    # print(results)
    #
    # """### Using a trained model to generate predictions on new data"""
    # ox = model2.predict(x_test)
    # print(ox)
