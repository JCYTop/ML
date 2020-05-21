# step 1: Preprocess The Data *********
# 处理数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("stduentscores.csv")
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

# step 2: 训练集使用简单线性回归模型来训练  *********

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# 使用线性模型进行训练数据接口 .fit
regressor = regressor.fit(X_train, Y_train)
# print(regressor)

# step 3: Predicting the Result *********
# 预测结果
Y_pred = regressor.predict(X_test)
print(Y_pred)

# step 4: Visualization
# 可视化
# .scatter 实现散点图
# .plot 实现预测函数
plt.scatter(X_train, Y_train, edgecolors="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
# plt.show()

# step 5: Test
# 测试结果可视化
plt.scatter(X_test, Y_test, edgecolors="yellow")
plt.plot(X_test, regressor.predict(X_test), color="black")
plt.show()
