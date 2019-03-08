import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed  = 2
def generateds():
    #基于seed产生随机数
    rdm = np.random.RandomState(seed)
    #随机数返回300行2列的矩阵，表示300组坐标点（X0，X1） 作为输入数据集
    X = rdm.randn(300,2)
    #从X这个300行2列的矩阵中取出一行，判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
    #作为输入数据集的标签（正确答案）
    Y_ = [[int(x0*x0+x1*x1)<2]for (x0,x1) in X]
    #遍历Y中的每一个元素，1赋值‘red’ 其余‘blue’,这样可视化显示
    Y_c =[['red'if l_y else 'blue']for y in Y_ for l_y in y]
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))  #squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    plt.show()
    return  X,Y_,Y_c
    # print(X)
    # print(Y_)
    # print(Y_c)
    #用plt.scatter 画出数据集X各行中第0行元素和第1行元素的点即各行的（x0,x1）,用
    #各行Y_c对应的值表示颜色
    # plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))  #squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    # plt.show()