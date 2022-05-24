---
layout: post
title:  "Model DSSM on Tensorflow"
date:   2016-06-21 15:20:52 +0800
categories: [AI, Tensorflow, Models]
---

Now with tensorflow installed, we now try to implement our first model on tensorflow. Instead of famous neural networks like LeNet, AlexNet, GoogleNet or ResNet, we choose a very simple but powerful model named
named **DSSM** (Deep Structured Semantic Models) for matching web search
queries and url based documents. The paper describing the model is published on CIKM'13 and available [here](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf). 

## DSSM model structure
![Illustration of the DSSM](https://raw.githubusercontent.com/v-liaha/v-liaha.github.io/master/assets/dssm.png)

### Model Input
The input of DSSM model are queries and documents mapped on so-called *n-gram* spaces instead of traditional word spaces. N-grams are defined as a sequence of n letters. A word, for example 'good', is firsted attached with hashtags on both sides to '#good#'. Then it is mapped into a list of 3-grams or trigrams: (*#go, goo, ood, od#*). As the number of characters are limited, the number of possible n-grams are limited as well, which is much smaller compared to number of words available. Typically as shown in Figure 1, the term vector with 500K words can be mapped to n-gram vectors sized only aroudn 30K. 

One query and several documents are input to the model at the same time. Only one of the documents, D1, is most related to the query, being the positive document. The other documents are all negative documents, that are not related to the query. A typical query/doc input is shown below as pairs of **N:X**, saying the **N**th n-gram appears **X** times in the query/doc. 

~~~
46238:1 24108:1 24016:1 5618:1 8818:1
~~~

### Neural Network
There are 3 fully connected (FC) layers in the network, with 300, 300, 128 neurons in each layer. Each input **x** is projected linearly by 
\\( Wx+b \\) and then activated non-linearly with \\( tanh/relu \\) functions to generate input for the next layer. 

First Layer: $$ l_1 = W_1x+b_1$$
Second Layer: $$ l_2 = f(W_2l_1+b_2) $$
Last Layer: $$ y = f(W_3l_2+b_3) $$

The output of the FC layers is a 128-length vector and fed to calculate cosine similarities. The cosine-similarity between the query and each document is calculated as:

$$ R(Q,D) = cosine(y_Q,y_D) = \frac{y_Q \cdot y_D}{\Vert y_Q \Vert \cdot \Vert y_D \Vert} $$

### Learning the DSSM

For *m* documents, there are *m* cosine similarity values, composing the logit vector. The score for each document is calculated as the posterior probability: 

$$ P(D \vert Q) = \frac{\gamma e^{R(Q,D)}}{\sum_{D'\in\mathbf{D}} \gamma e^{R(Q,D')}} $$

The loss function is finally defined as:

$$ L(\Lambda) = -log\prod_{(Q,D^+)} P(D^+\vert Q)$$

## Tensorflow Implementation

### Import Tensorflow

~~~python
import tensorflow as tf
~~~

### Input Batching and Sparsifying

To fully utilize the GPU capability, we feed the model with Q and D in batches of size **BS**. The original input vector [TRIGRAM_D] is now a matrix with size shaped [BS, TRIGRAM_D]. TRIGRAM_D is the total number of trigrams appear in all queries and documents.

Another problem is that the input matrix is very sparse. We find that 80% of the queries can be composed of less than 30 trigrams, which makes most of the input matrix values zero. 

Tensorflow supports sparse placeholders, which are used to hold the input tensors:

~~~python
query_batch = tf.sparse_placeholder(tf.float32, 
                                    shape=[None,TRIGRAM_D], 
                                    name='QueryBatch')
doc_batch = tf.sparse_placeholder(tf.float32, 
                                    shape=[None, TRIGRAM_D], 
                                    name='DocBatch')
~~~

### Initialize Weight and Bias
Then the weights and biases are specified for each layer, here only layer 1 is presented. Weights and biases are initialized in the uniform distribution as decribed in the paper.

~~~python
# L1_N = 300
l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], 
                                        -l1_par_range, 
                                        l1_par_range))
bias1 = tf.Variable(tf.random_uniform([L1_N], 
                                       -l1_par_range, 
                                       l1_par_range))
~~~

### Define FC Layer Operations

Generate the activations (output of the layer) using the sparse-dense matrix multiplication operator and the relu activation function:

~~~python
query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

query_l1_out = tf.nn.relu(query_l1)
doc_l1_out = tf.nn.relu(doc_l1)
~~~

### Cosine Similarity

After L2 and L3, we get **y**, the output of 3 FC layers, and feed it to cosine-similarity. Here we only caculate \\( y_Q \\) once and duplicate it for the calculation of cosine-similarity.

~~~python
# NEG is the number of negative documents
query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), 
                     [NEG + 1, 1])
doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
norm_prod = tf.mul(query_norm, doc_norm)

cos_sim_raw = tf.truediv(prod, norm_prod)
cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * Gamma
~~~

### Loss

Finally, the loss is calculated and averaged by the batch size. 

~~~python
prob = tf.nn.softmax((cos_sim))
hit_prob = tf.slice(prob, [0, 0], [-1, 1])
loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
~~~

### Training in One Line!
Training using gradient descent:

~~~python
train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
~~~

### Actually Run DSSM Model

Running Tensorflow with a session:

~~~python
# Allow GPU to allocate memory dynamically 
# instead of taking all memory at once in 
# the beginning
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(FLAGS.max_steps):
        sess.run(train_step, feed_dict={query_batch : ...
                                        doc_batch   : ...}})
~~~

That's it! The code structure is very clear and we don't even need to compose the training part. Tensorflow is capable of constructing the training operation graph automatically. The full code is publicly available in my [github page](https://github.com/v-liaha/tensorflow/blob/r0.9/tensorflow/models/dssm/dssm.py).

## Tensorboard Visualization

We have worked some tricks to change the model a little bit, and one of the advantages of tensorflow is that it provides a visualization called tensorboard to visualize the network you created. I may provide more information on it, but now I'm just providing the visualization of our DSSM network structure:

![Tensorboard Visualization](https://raw.githubusercontent.com/v-liaha/v-liaha.github.io/master/assets/dssm-tensorboard.png)
