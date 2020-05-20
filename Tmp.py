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
