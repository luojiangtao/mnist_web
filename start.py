import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from predict import convolutional

# webapp
app = Flask(__name__)


'''
request.json
一维数组，784个特征
[255, 161, 0, 0,....]
'''
@app.route('/api/mnist', methods=['POST'])
def mnist():
    #标准化数据
    # input = np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    # input = np.array(request.json, dtype=np.uint8).reshape(1, 784)
    # 把白底黑字的图片转换为黑底白字
    input = abs((255 - np.array(request.json, dtype=np.uint8)).reshape(1, 784))

    # 保存图片
    # import scipy.misc
    # image = input.reshape(28, 28)
    # scipy.misc.imsave('a.jpg', image)
    #标准化数据
    # input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    print(input)
    #一维数组，输出10个预测概率
    output = convolutional(input)
    '''
    {"results":[
        [0.0005708700628019869,0.010075394995510578,0.8699323534965515,0.0013963828096166253,0.028609132394194603,0.006814470514655113,0.06850877404212952,0.006337625440210104,0.004338784143328667,0.003416265593841672],
        [5.7194258261006325e-05,0.0006196154863573611,0.9920960664749146,0.000495785498060286,1.5396590242744423e-05,0.002464226447045803,0.00023624727327842265,0.0021845928858965635,0.0004759470175486058,0.001354911015368998]
    ]}
    '''
    return jsonify(results=output)
    # return jsonify(results=[1])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run(host='localhost')
