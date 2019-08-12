import os

import tensorflow as tf
from PIL import Image
import keras
from keras import layers



model = keras.Sequential()

model.add(layers.Dense(32, activation='relu'))

labels = ["cat", "dog", "car"]
index = {"cat": 0, "dog": 1, "car": 2}

pwd = "F:/Data/dogs-vs-cats/train/"



def DataGet():
    pwd = "F:/Data/dogs-vs-cats/train"
    classes = {"cat": 0, "dog": 1, "car": 2}

    print("====loading====from " + pwd)
    for trainpwdItem in os.listdir(pwd):
        childPath = pwd + "/" + trainpwdItem
        print("====loading====from " + childPath)
        for imageItem in os.listdir(childPath):
            imageItemPath = childPath + "/" + imageItem
            img = Image.open(imageItemPath)
            img_raw = img.tobytes()


def TFrecordMaker():
    pwd = "F:/Data/dogs-vs-cats/train"
    classes = {"cat": 0, "dog": 1, "car": 2}
    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    print("====loading====from " + pwd)
    for trainpwdItem in os.listdir(pwd):
        childPath = pwd + "/" + trainpwdItem
        print("====loading====from " + childPath)
        for imageItem in os.listdir(childPath):
            imageItemPath = childPath + "/" + imageItem
            img = Image.open(imageItemPath)
            img = img.resize((255, 255), Image.ANTIALIAS)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[classes.get(trainpwdItem)])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


def TFrecordOnceParse(example):
    dics = {
        "label": tf.FixedLenFeature([], tf.int64),
        "img_raw": tf.FixedLenFeature([], tf.string)
    }

    parsed_example = tf.parse_single_example(serialized=example, features=dics)
    image = tf.decode_raw(parsed_example['img_raw'], out_type=tf.uint8)
    print(tf.shape(image))
    image = tf.reshape(image, [256, -1])
    label = parsed_example['label']

    return image, label




if __name__ == '__main__':
    #TFrecordMaker()
    dataset = tf.data.TFRecordDataset("train.tfrecords")
    dataset = dataset.map(TFrecordOnceParse)
    # dataset = dataset.shuffle(10)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(10):
            img, label = sess.run(fetches=next_element)
            print(len(img))



