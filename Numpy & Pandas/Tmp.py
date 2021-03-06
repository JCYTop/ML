import numpy as np

array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)
print(array.ndim)
array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
print(array)
print(array.dtype)
print(np.arange(12))
print(np.arange(12).reshape((3, 4)))
print(np.linspace(1, 10, 20))

a = np.array([2, 23, 4])
print(a)

a = np.array([10, 20, 30, 40])
print(a)
b = np.arange(4)
print(b)
print(a - b)
print(b < 2)

print("---------------------------")

a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))
print(a)
print(a.T)
print(b)
c = a * b
c_dot = np.dot(a, b)

print(c)
print(c_dot)

a = np.random.random(2)
print(np.min(a))

print(" ")
print("pandas-------------------------------------")
import pandas as pd

dates = pd.date_range("20200605", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
# print(df['A'])
# print(df.A)
print(df.loc['20200608', 'A':])

print(" ")
print(" -------------------------------------")
import matplotlib.pyplot as plt

# Series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()

# DataFrame
data = pd.DataFrame(np.random.randn(1000, 4), columns=list("ABsD"))

data.plot()
plt.show()
