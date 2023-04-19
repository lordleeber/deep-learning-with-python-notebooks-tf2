

"""### Further experiments

### Wrapping up

## Classifying newswires: A multiclass classification example

### The Reuters dataset

**Loading the Reuters dataset**
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


def build_model1():
    """### Building your model
    **Model definition**
    """
    model1 = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
    ])

    """**Compiling the model**"""
    model1.compile(optimizer="rmsprop",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    return model1


def build_model2():
    model2 = keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(46, activation="softmax")
    ])

    model2.compile(optimizer="rmsprop",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    return model2


def build_model3():
    model3 = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(46, activation="softmax")
    ])

    model3.compile(optimizer="rmsprop",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    return model3


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print(len(train_data))
    print(len(test_data))
    print(train_data[10])

    """**Decoding newswires back to text**"""
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
    print(train_labels[10])

    """### Preparing the data
    **Encoding the input data**
    """
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    """**Encoding the labels**"""
    y_train = to_one_hot(train_labels)
    y_test = to_one_hot(test_labels)

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    """### Validating your approach
    **Setting aside a validation set**
    """
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    """**Training the model**"""
    model1 = build_model1()
    history = model1.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

    """**Plotting the training and validation loss**"""

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    """**Plotting the training and validation accuracy**"""

    plt.clf()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    """**Retraining a model from scratch**"""
    model2 = build_model2()
    model2.fit(x_train,
               y_train,
               epochs=9,
               batch_size=512)
    results = model2.evaluate(x_test, y_test)
    print(results)

    import copy
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(test_labels) == np.array(test_labels_copy)
    hits_array.mean()

    """### Generating predictions on new data"""

    predictions = model2.predict(x_test)

    print(predictions[0].shape)

    np.sum(predictions[0])

    np.argmax(predictions[0])

    """### A different way to handle the labels and the loss"""
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    model2.compile(optimizer="rmsprop",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    """### The importance of having sufficiently large intermediate layers
    **A model with an information bottleneck**
    """
    model3 = build_model3()
    model3.fit(partial_x_train,
               partial_y_train,
               epochs=20,
               batch_size=128,
               validation_data=(x_val, y_val))

