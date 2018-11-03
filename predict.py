import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from PIL import Image
import model



def convolutional(input):
    print(tf.__version__)

    x = tf.placeholder("float", [None, 784])
    sess = tf.Session()


    with tf.variable_scope("convolutional"):
        keep_prob = tf.placeholder("float")
        y2, variables = model.convolutional(x, keep_prob)
    saver = tf.train.Saver(variables)
    saver.restore(sess, "./model/model.ckpt")
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


if __name__ == "__main__":
    '''
    request.json
    一维数组，784个特征
    [255, 161, 0, 0,....]
    '''

    I = Image.open('./test_image/0.jpg')
    # I.show()
    # I.save('./save.png')
    I_array = np.array(I)
    # print (I_array.shape)

    # input = ((255 - np.array(I_array, dtype=np.uint8)) / 255.0).reshape(1, 784)
    input = I_array.reshape(1, 784)
    #一维数组，输出10个预测概率
    output2 = convolutional(input)
    print(output2)
'''
{"results":[
    [5.7194258261006325e-05,0.0006196154863573611,0.9920960664749146,0.000495785498060286,1.5396590242744423e-05,0.002464226447045803,0.00023624727327842265,0.0021845928858965635,0.0004759470175486058,0.001354911015368998]
]}
    '''
