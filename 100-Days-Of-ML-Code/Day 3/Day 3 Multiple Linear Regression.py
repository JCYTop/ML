# 多元线性回归
# step 1: Data Preprocessing
# 数据预处理

# 导入库
import pandas as pd
import numpy as np

# 导入数据集
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, :4].values

# 将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
state = np.unique(X[:, 3])
onehotencoder = OneHotEncoder(categories="auto")
X = onehotencoder.fit_transform(X)
# 躲避虚拟变量陷阱
X = X[:, 1:]
print(X)
# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# step 2: 在训练集上训练多元线性回归模型
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# step 2: 在测试集上预测结果
# y_pred = regressor.predict(X_test)
