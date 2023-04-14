
"""## Data representations for neural networks

### Scalars (rank-0 tensors)
"""

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

x = np.array(12)
print(x)

print(x.ndim)

"""### Vectors (rank-1 tensors)"""

x = np.array([12, 3, 6, 14, 7])
print(x)

print(x.ndim)

"""### Matrices (rank-2 tensors)"""

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)

"""### Rank-3 and higher-rank tensors"""

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x.ndim)

"""### Key attributes"""

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)

print(train_images.shape)

print(train_images.dtype)

"""**Displaying the fourth digit**"""

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print(train_labels[4])

"""### Manipulating tensors in NumPy"""

my_slice = train_images[10:100]
print(my_slice.shape)

my_slice = train_images[10:100, :, :]
print(my_slice.shape)

my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

my_slice = train_images[:, 14:, 14:]

my_slice1 = train_images[:, 7:-7, 7:-7]

"""### The notion of data batches"""

batch = train_images[:128]

batch1 = train_images[128:256]

n = 3
batch2 = train_images[128 * n:128 * (n + 1)]


"""### Tensor reshaping"""

train_images = train_images.reshape((60000, 28 * 28))

x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])
print(x.shape)

x = x.reshape((6, 1))
print(x)

x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)
