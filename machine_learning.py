# Library Imports

# [[file:~/delta-hacks-ML-workshop/README.org::*Library Imports][Library Imports:1]]
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
print (tf.__version__)
# Library Imports:1 ends here

# Unknowns and Knowns

# Here's the unknowns that the algorithm will ‘learn’.

# [[file:~/delta-hacks-ML-workshop/README.org::*Unknowns and Knowns][Unknowns and Knowns:1]]
𝒷 = tf.Variable([.3], tf.float32)
𝓌 = tf.Variable([-.3], tf.float32)
# Unknowns and Knowns:1 ends here



# #+RESULTS:

# Here's the date we will provide samples of.

# [[file:~/delta-hacks-ML-workshop/README.org::*Unknowns and Knowns][Unknowns and Knowns:2]]
x =  tf.placeholder(tf.float32)
y =  tf.placeholder(tf.float32)
# Unknowns and Knowns:2 ends here

# Training Data

#   The example inputs and outputs from before.

# [[file:~/delta-hacks-ML-workshop/README.org::*Training Data][Training Data:1]]
# x_data = [-3, -2, -1, 0, 1, 2, 3]
# y_data = [12, 11, 10, 9, 8, 7, 6]
x_data = [4.0, 0.0, 12.0]
y_data = [5.0, 9, -3]
# Training Data:1 ends here

# Math: Error & loss functions, optimisation, and initialising the model

# [[file:~/delta-hacks-ML-workshop/README.org::*Math: Error & loss functions, optimisation, and initialising the model][Math: Error & loss functions, optimisation, and initialising the model:1]]
learning_rate = 0.001

model = 𝓌 * x + 𝒷
delta = tf.square(model - y) # error function
loss  = tf.reduce_sum(delta)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()
# Math: Error & loss functions, optimisation, and initialising the model:1 ends here

# Actually Train!


# [[file:~/delta-hacks-ML-workshop/README.org::*Actually Train!][Actually Train!:1]]
with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        feed_dict_batch = {x: x_data, y: y_data}
        sess.run(optimizer, feed_dict = feed_dict_batch)

    approx_w, approx_b = sess.run([𝓌, 𝒷])
    print("𝓌 ≈", approx_w, "and 𝒷 ≈", approx_b)
# Actually Train!:1 ends here
