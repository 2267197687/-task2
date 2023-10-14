import random
import numpy as np


class Network(object):
    
    #生成对应权重矩阵，偏置向量
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    #前馈，model
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    #优化器主体框架
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        
        self.list_test_data = list(test_data)
        n_test_data = len(self.list_test_data)
        if test_data:
            
            n_test = n_test_data
            
        list_training_data = list(training_data)
        n = len(list_training_data)
        for j in range(epochs):
            random.shuffle(list_training_data)#打乱数据集
            
            list_mini_batches = [
                list_training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]#划分batch
            mini_batches = zip(*list_mini_batches)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)#更新参数
            if test_data:
                print("Epoch {0}: {1} / {2}".format(#评估模型
                    j, self.evaluate(), n_test))
            else:
                print("Epoch {0} complete".format(j))
            print(self.runningloss)

    #以batch为单位更新参数
    def update_mini_batch(self, mini_batch, eta):
       
        nabla_b = [np.random.normal(0, 1, b.shape) for b in self.biases]#生成对应零矩阵，初始化
        nabla_w = [np.random.normal(0, 1, w.shape) for w in self.weights]
        for x, y in mini_batch:
            #反向传播
            delta_nabla_b, delta_nabla_w ,loss= self.backprop(x, y)
            #对应偏导数
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss
            #更新
        self.weights = [w - (eta / len(list(mini_batch))) * nw for w, nw in zip(
            self.weights, nabla_w)]
        self.biases = [b - (eta / len(list(mini_batch))) * nb for b, nb in zip(
            self.biases, nabla_b)]
        self.runningloss = loss
    #反向传播
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]  
        zs = [] 
        #记录输入输出 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #从输出层开始计算梯度

        loss = self.cost_derivative(activations[-1], y).sum()
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])#关于z
        nabla_b[-1] = delta#关于b
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())#关于w
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w, loss)

    #评估
    def evaluate(self):
        
        
        test_results = [(np.argmax(self.feedforward(x))#预测最有可能的类别
                          , y)
                        for x, y in self.list_test_data]
        list_test_results = list(test_results)
        
        return sum(int(x == y) for x, y in list_test_results)

    #损失函数
    def cost_derivative(self, output_activations, y):
        
        return (output_activations - y)


#sigmoid激活函数
def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-z))

#导数
def sigmoid_prime(z):
    
    return sigmoid(z) * (1 - sigmoid(z))