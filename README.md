# Table of Contents

1.  [Goal](#Goal)
2.  [Sample Training Data](#Sample-Training-Data)
3.  [Installations](#Installations)
4.  [Library Imports](#Library-Imports)
5.  [Unknowns and Knowns](#Unknowns-and-Knowns)
6.  [Training Data](#Training-Data)
7.  [Math: Error & loss functions, optimisation, and initialising the model](#Math-Error-loss-functions-optimisation-and-initialising-the-model)
8.  [Actually Train!](#Actually-Train)
9.  [Approximations of 𝓌 and 𝒷](#Approximations-of-𝓌-and-𝒷)
10. [References](#References)

-   These notes and slides were based on a [hackernoon article](https://hackernoon.com/build-your-first-tensorflow-model-in-5-minutes-77237e3cf76d).
-   The slides and code where written using [literate programming](https://github.com/alhassy/emacs.d#what-does-literate-programming-look-like).
-   Below are the notes which can be seen as live  [slides](https://alhassy.github.io/delta-hacks-ML-workshop/machine-learning.html) &#x2014;or see the [website](https://alhassy.github.io/delta-hacks-ML-workshop/).

Next steps: Read “[How to get started with Machine Learning in about 10 minutes](https://www.freecodecamp.org/news/how-to-get-started-with-machine-learning-in-less-than-10-minutes-b5ea68462d23/)”
(•̀ᴗ•́)و

&#x2026;or skip ahead and see [our notebook](https://colab.research.google.com/drive/1CVn0hRCP6-Bfc3IbMECEyKPOF7qm90LX0)


<a id="Goal"></a>

# Goal

-   Given `x`, compute `y = 𝓌*x + 𝒷`.

-   Wait, what are 𝓌 and 𝒷?

-   We'll “train our model” by giving it
    examples of `x` inputs and `y` outputs.

**Idea**: Given examples `(x,y)`, and an `x`, find a ‘suitable’ `y`.


<a id="Sample-Training-Data"></a>

# Sample Training Data

“Supervised machine learning” is when we know the output `y` for some inputs `x`.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">x</th>
<th scope="col" class="org-right">y</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">-3</td>
<td class="org-right">12</td>
</tr>


<tr>
<td class="org-right">-2</td>
<td class="org-right">11</td>
</tr>


<tr>
<td class="org-right">-1</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-right">0</td>
<td class="org-right">9</td>
</tr>


<tr>
<td class="org-right">1</td>
<td class="org-right">8</td>
</tr>


<tr>
<td class="org-right">2</td>
<td class="org-right">7</td>
</tr>


<tr>
<td class="org-right">3</td>
<td class="org-right">6</td>
</tr>
</tbody>
</table>

⇒ Questions: What is `y` when `x = 5`?

Goal: **Get the model to *learn* that 𝓌 is -1 and 𝒷 is 9**!

**Real world:** A machine ‘learns’ what is a cat by being exposed to many pictures
of cats!


<a id="Installations"></a>

# Installations

Two possible directions …

-   <span class="underline">On your own machine</span>

        pip3 install python3
        pip3 install tensorflow

-   <span class="underline">Using a website</span>
    <https://colab.research.google.com/notebooks/intro.ipynb>


<a id="Library-Imports"></a>

# Library Imports

    # import tensorflow as tf
    import tensorflow.compat.v1 as tf
    import numpy as np

    tf.disable_v2_behavior()
    print (tf.__version__)

    2.1.0

Alternatively, one could install an older version of tensorflow `pip install
tensorflow==1.4`.


<a id="Unknowns-and-Knowns"></a>

# Unknowns and Knowns

Here's the unknowns that the algorithm will ‘learn’.

    𝒷 = tf.Variable([.3], tf.float32)
    𝓌 = tf.Variable([-.3], tf.float32)

Here's the date we will provide samples of.

    x =  tf.placeholder(tf.float32)
    y =  tf.placeholder(tf.float32)


<a id="Training-Data"></a>

# Training Data

The example inputs and outputs from before.

    # x_data = [-3, -2, -1, 0, 1, 2, 3]
    # y_data = [12, 11, 10, 9, 8, 7, 6]
    x_data = [4.0, 0.0, 12.0]
    y_data = [5.0, 9, -3]


<a id="Math-Error-loss-functions-optimisation-and-initialising-the-model"></a>

# Math: Error & loss functions, optimisation, and initialising the model

    learning_rate = 0.001

    model = 𝓌 * x + 𝒷
    delta = tf.square(model - y) # error function
    loss  = tf.reduce_sum(delta)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">This is where human creativity comes in!</td>
</tr>
</tbody>
</table>


<a id="Actually-Train"></a>

# Actually Train!

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            feed_dict_batch = {x: x_data, y: y_data}
            sess.run(optimizer, feed_dict = feed_dict_batch)

        approx_w, approx_b = sess.run([𝓌, 𝒷])
        print("𝓌 ≈", approx_w, "and 𝒷 ≈", approx_b)


<a id="Approximations-of-𝓌-and-𝒷"></a>

# Approximations of 𝓌 and 𝒷

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">Iterations</th>
<th scope="col" class="org-right">𝓌</th>
<th scope="col" class="org-right">𝒷</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">1</td>
<td class="org-right">-0.2456</td>
<td class="org-right">0.3298</td>
</tr>


<tr>
<td class="org-right">100</td>
<td class="org-right">-0.3364</td>
<td class="org-right">2.4222</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">-0.9454</td>
<td class="org-right">8.45914</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">-0.9999</td>
<td class="org-right">8.99983</td>
</tr>
</tbody>
</table>


<a id="References"></a>

# References

-   These notes and slides were based on a [hackernoon article](https://hackernoon.com/build-your-first-tensorflow-model-in-5-minutes-77237e3cf76d).
-   The slides and code where written using [literate programming](https://github.com/alhassy/emacs.d#what-does-literate-programming-look-like).
-   Next steps: Read “[How to get started with Machine Learning in about 10 minutes](https://www.freecodecamp.org/news/how-to-get-started-with-machine-learning-in-less-than-10-minutes-b5ea68462d23/)”

    &#x2026;or skip ahead and see [our notebook](https://colab.research.google.com/drive/1CVn0hRCP6-Bfc3IbMECEyKPOF7qm90LX0)

(•̀ᴗ•́)و

⇒ [Github Repo](https://github.com/alhassy/delta-hacks-ML-workshop) ⇐
