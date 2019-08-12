import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    file = "F:/Data/MNIST"

    mnist = input_data.read_data_sets(file, one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    output = tf.placeholder(tf.float32, shape=[None, 10])

    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    session = tf.InteractiveSession()

    session.run(tf.global_variables_initializer())

    y = tf.nn.softmax(tf.matmul(x, w) + b)

    cross_entropy = -tf.reduce_sum(output*tf.log(y))


    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], output:batch[1]})


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x:mnist.test.images, output:mnist.test.labels}))