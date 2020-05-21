# step 1: Importing the libraries *********
# 导入数据库
# NumPy 包含数学计算函数
# Pandas 用于导入和管理数据集

import numpy as np
import pandas as pd

# step 2: Importing dataset *********
# 导入使用数据

# ../ 表示当前文件所在的目录的上一级目录
# ./ 表示当前文件所在的目录(可以省略)
# / 表示当前站点的根目录(域名映射的硬盘目录)

dataset = pd.read_csv("./Data.csv")
# .iloc[行，列]
# .iloc 获取数据表中指定的行 列
X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 3]
# print(Y)

# step 3: Handle NaN(the missing) data *********
# from sklearn.preprocessing import Imputer --->>> ***obsolete***
# 替换成 from sklearn.impute import SimpleImputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# .fit查找替换输入范围中每一要替换的值
imputer = imputer.fit(X[:, 1:3])
# .transform 转移数据使用
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# strp 4: Encoding categorical data *********
# 解析分类数据

# LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(labelencoder_X.fit_transform(X[:, 0]))
# OneHotEncoder(categorical_features = [0]) --->>> ***obsolete***
# 替换成 OneHotEncoder(categories='auto')

onehotencoder = OneHotEncoder(categories='auto')
# print(X)
X = onehotencoder.fit_transform(X)
# print(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# print(Y)

# Step 5: Splitting the datasets into training sets and Test sets *********
# 拆分数据分配为训练数据和测试数据
# from sklearn.cross_validation import train_test_split --->>> ***obsolete***
# 替换为 from sklearn.model_selection import  train_test_split

from sklearn.model_selection import train_test_split

# train_test_split 随即划分训练集和测试集
# train_data 所要划分的样本特征集
# train_target 所要划分的样本结果
# test_size 样本占比，如果是整数的话就是样本的数量
# random_state 随机数的种子
# * 多参
# ** 字典传入
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# step 6: Feature Scaling *********
# 特征量化

from sklearn.prepropipcessing import StandardScaler

sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print("-------")
print(X_test)
