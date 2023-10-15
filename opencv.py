import cv2
import numpy as np
from keras.datasets import mnist


if __name__=='__main__':

    #使用Keras载入训练数据
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    #变换数据的形状并归一化
    train_images=train_images.reshape(train_images.shape[0],-1)#(60000, 28, 28) -> (60000, 784)
    train_images=train_images.astype('float32')/255#变为float32，像素值为0~255，可以放缩到0~1

    test_images=test_images.reshape(test_images.shape[0],-1)
    test_images=test_images.astype('float32')/255
    
    #将标签数据转为int32 并且形状为(60000,1)
    train_labels=train_labels.astype(np.int32)
    test_labels=test_labels.astype(np.int32)
    train_labels=train_labels.reshape(-1,1)
    test_labels=test_labels.reshape(-1,1)

    #创建svm模型
    svm = cv2.ml.SVM_create()
    #设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    #设置核函数kernel
    svm.setKernel(cv2.ml.SVM_POLY)
    #单个样本的影响程度
    svm.setGamma(3)
    #kernel核函数的次数
    svm.setDegree(3)
    #设置迭代终止条件
    svm.setTermCriteria((cv2.TermCriteria_MAX_ITER,300,1e-3))
    #训练
    svm.train(train_images,cv2.ml.ROW_SAMPLE,train_labels)
    svm.save('mnist_svm.xml')#保存在该文件

    #在测试数据上计算准确率
    #进行模型准确率的测试 结果是一个元组 第一个值为数据1的结果
    test_pre=svm.predict(test_images)
    test_ret=test_pre[1]#取出预测标签

    #计算准确率
    test_ret=test_ret.reshape(-1,)
    test_labels=test_labels.reshape(-1,)
    test_sum=(test_ret==test_labels)
    acc=test_sum.mean()#将T和F转换为0和1
    print(acc)


