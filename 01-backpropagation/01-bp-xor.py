import numpy as np


# 激活函数和它的导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 输入数据
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# 输出数据
y = np.array([[0],
              [1],
              [1],
              [0]])

# 设置随机种子以获得可重复的结果
np.random.seed(1)

# 学习率
learning_rate = 0.9

# 初始化权重
synapse_0 = 2 * np.random.random((3, 4)) - 1
synapse_1 = 2 * np.random.random((4, 1)) - 1

# 训练循环
for j in range(60000):

    # 前向传播
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    # 计算误差
    layer_2_error = y - layer_2

    if (j % 1000) == 0:
        print("Error:" + str(np.mean(np.abs(layer_2_error))))

    # 反向传播
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # 更新权重
    synapse_0 += learning_rate * layer_0.T.dot(layer_1_delta)
    synapse_1 += learning_rate * layer_1.T.dot(layer_2_delta)

print("Output After Training:")
print(layer_2)


# 修改预测函数
def predict(input_data):
    layer_1 = sigmoid(np.dot(input_data, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    # 将预测结果转换为整数二元输出
    return np.round(layer_2).astype(int)


# 对训练数据进行预测
print("\n训练数据的预测结果：")
for i in range(len(X)):
    prediction = predict(X[i])
    print(f"输入: {X[i]}, 预测输出: {prediction[0]}, 实际输出: {y[i][0]}")

# 测试新的输入数据
new_input = np.array([1, 1, 0])
prediction = predict(new_input)
print(f"\n新输入 [1,1,0] 的预测结果: {prediction[0]}")
