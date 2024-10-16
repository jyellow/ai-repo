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
hidden_size = 4
Wxh = 2 * np.random.random((3, hidden_size)) - 1  # 输入到隐藏层的权重
Whh = 2 * np.random.random((hidden_size, hidden_size)) - 1  # 隐藏层到隐藏层的权重
Why = 2 * np.random.random((hidden_size, 1)) - 1  # 隐藏层到输出的权重

# 训练循环
for j in range(60000):
    # 前向传播
    h = np.zeros((1, hidden_size))  # 初始化隐藏状态
    layer_2_error = 0
    for i in range(len(X)):
        x = X[i:i + 1]
        h = sigmoid(np.dot(x, Wxh) + np.dot(h, Whh))
        y_pred = sigmoid(np.dot(h, Why))
        layer_2_error += np.abs(y[i] - y_pred)

    if (j % 1000) == 0:
        print("Error:" + str(np.mean(layer_2_error)))

    # 反向传播
    dWhy = np.zeros_like(Why)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dh_next = np.zeros_like(h)

    h_prev = np.zeros((1, hidden_size))  # 初始化 h_prev
    for i in reversed(range(len(X))):
        x = X[i:i + 1]
        h = sigmoid(np.dot(x, Wxh) + np.dot(h_prev, Whh))
        y_pred = sigmoid(np.dot(h, Why))

        dy = y[i] - y_pred
        dWhy += np.dot(h.T, dy)

        dh = np.dot(dy, Why.T) + dh_next
        dh_raw = dh * sigmoid_derivative(h)

        dWxh += np.dot(x.T, dh_raw)
        dWhh += np.dot(h_prev.T, dh_raw)

        dh_next = np.dot(dh_raw, Whh.T)
        
        h_prev = h  # 更新 h_prev 为当前的 h

    # 更新权重
    Wxh += learning_rate * dWxh
    Whh += learning_rate * dWhh
    Why += learning_rate * dWhy

print("训练后的输出:")
h = np.zeros((1, hidden_size))
for i in range(len(X)):
    x = X[i:i + 1]
    h = sigmoid(np.dot(x, Wxh) + np.dot(h, Whh))
    y_pred = sigmoid(np.dot(h, Why))
    print(y_pred)


# 修改预测函数
def predict(input_data):
    h = np.zeros((1, hidden_size))
    for i in range(len(input_data)):
        x = input_data[i:i + 1]
        h = sigmoid(np.dot(x, Wxh) + np.dot(h, Whh))
    y_pred = sigmoid(np.dot(h, Why))
    return np.round(y_pred).astype(int)


# 对训练数据进行预测
print("\n训练数据的预测结果：")
for i in range(len(X)):
    prediction = predict(X[i:i + 1])
    print(f"输入: {X[i]}, 预测输出: {prediction[0][0]}, 实际输出: {y[i][0]}")

# 测试新的输入数据
new_input = np.array([[1, 1, 0]])
prediction = predict(new_input)
print(f"\n新输入 [1,1,0] 的预测结果: {prediction[0][0]}")
