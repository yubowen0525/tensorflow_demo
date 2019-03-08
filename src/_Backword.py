import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _Generateds
import _Forward


BATCH_SIZE = 30
STEPS = 40000
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.99
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X,Y_,Y_c=_Generateds.generateds()

    y=_Forward.forword(x,REGULARIZER)

    global_step =tf.Variable(0,trainable=False)
    # 定义指数下降学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               300/BATCH_SIZE, LEARNING_RATE_DECAY,
                                               staircase=True)
    #定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    #定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
    # train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
                print('After %d steps , loss_mse_v is : %f' % (i, loss_mse_v))

            # XX在-3到3 之间以布长为0.01，yy在-3到3之间以步长为0.01 ，生成二维网络坐标点
            # mgrid  制造网格
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        # 将xx,yy拉直，并合并成一个2列的矩阵，得到一个网络坐标点的集合
        # np.ravel()降成一维 行优先
        grid = np.c_[xx.ravel(), yy.ravel()]
        # 将网络坐标点喂入神经网络，probs为输出
        probes = sess.run(y, feed_dict={x: grid})
        # probs的shape调整成xx的样子
        probes = probes.reshape(xx.shape)

        plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
        plt.contour(xx, yy, probes, levels=[.5])
        plt.show()


if __name__ == '__main__':
    backward()
