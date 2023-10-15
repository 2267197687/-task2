import cv2
import numpy as np

if __name__=='__main__':
    #读取图片
    img=cv2.imread('2437.jpg',0)#0表示灰度模式
    img_sw=img.copy()

    #将数据类型由uint8转为float32
    img=img.astype(np.float32)
    #图片形状由(28,28)转为(784,)
    img=img.reshape(-1,)
    #增加一个维度变为(1,784)
    img=img.reshape(1,-1)
    #图片数据归一化
    img=img/255

    #载入svm模型
    svm=cv2.ml.SVM_load('mnist_svm.xml')
    #进行预测
    img_pre=svm.predict(img)
    print(img_pre[1])

    cv2.imshow('test',img_sw)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'): #输入q退出
            break
    cv2.destroyAllWindows()
