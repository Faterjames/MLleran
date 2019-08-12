import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
if __name__ == '__main__':

    file = "F:/Data/MNIST"

    mnist = input_data.read_data_sets(file, one_hot=True)

    batch_size = 50
    n_batch =  mnist.train.num_examples

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    w1 = tf.Variable(tf.random_normal([784, 100]))
    b1 = tf.Variable(tf.random_normal([1, 100]))
    r1 = tf.matmul(x, w1) + b1

    L1 = tf.nn.tanh(r1)

    w2 = tf.Variable(tf.random_normal([100, 100]))
    b2 = tf.Variable(tf.random_normal([1, 100]))
    r2 = tf.matmul(L1, w2) + b2

    L2 = tf.nn.tanh(r2)

    w3 = tf.Variable(tf.random_normal([100, 10]))
    b3 = tf.Variable(tf.random_normal([1, 10]))
    r3 = tf.matmul(L2, w3) + b3
    prediction = tf.nn.tanh(r3)

    wt1 = tf.Variable(tf.random_normal([1, 9]))
    rt1 = tf.matmul(x, wt1)
    wt2 = tf.Variable([16, 16, 37, 16, 4, 16, 1, 16, 16])
    rt2 = tf.matmul(wt1, wt2)

    loss = tf.reduce_mean(tf.square(y - prediction))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(500):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))


