"""### Real-world examples of data tensors

### Vector data

### Timeseries data or sequence data

### Image data

### Video data

## The gears of neural networks: tensor operations

### Element-wise operations
"""

import time
import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)
print("0 Took: {0:.2f} s".format(time.time() - t0))

t1 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("1 Took: {0:.2f} s".format(time.time() - t1))


"""### Broadcasting"""

X = np.random.random((32, 10))
y = np.random.random((10,))

y = np.expand_dims(y, axis=0)

Y = np.concatenate([y] * 32, axis=0)


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)


"""### Tensor product"""

x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# def naive_matrix_vector_dot(x, y):
#     assert len(x.shape) == 2
#     assert len(y.shape) == 1
#     assert x.shape[1] == y.shape[0]
#     z = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             z[i] += x[i, j] * y[j]
#     return z


def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

