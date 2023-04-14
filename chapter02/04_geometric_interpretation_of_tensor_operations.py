
"""### Geometric interpretation of tensor operations

### A geometric interpretation of deep learning

## The engine of neural networks: gradient-based optimization

### What's a derivative?

### Derivative of a tensor operation: the gradient

### Stochastic gradient descent

### Chaining derivatives: The Backpropagation algorithm

#### The chain rule

#### Automatic differentiation with computation graphs

#### The gradient tape in TensorFlow
"""

import tensorflow as tf


x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
    print(x)
    print(y)
grad_of_y_wrt_x = tape.gradient(y, x)


x1 = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x1 + 3
    print(x1)
    print(y)
grad_of_y_wrt_x1 = tape.gradient(y, x1)


W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])


