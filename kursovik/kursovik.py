from ast import Import
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
import timeit

#Linear Search
def LinearSearch(array, x):
    k = 0
    for i in range(0, len(array)):
        if x == array[i]:
            return True
    return False

#Jump Search
def JumpSearch(arr, x):
    n = len(arr)
    if n == 0:
        return False  
    step = int(math.sqrt(n))

    prev = 0
    while arr[min(step, n) - 1] < x:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return False  
    while prev < min(step, n):
        if arr[prev] == x:
            return True 
        prev += 1

    return False


#Binary Search
def BinarySearch(array, x):
    left = 0
    right = len(array) - 1
    res = False
    while (left <= right and not res):
        mid = (left + right) // 2
        b = array[mid]
        if x == b:
            res = True 
            return res 
        if (b > x):
            right = mid - 1
        else: 
            left = mid + 1
    return res

#Interpolation Search
def InterpolationSearch(array, x):
    l = 0
    r = len(array) - 1
    res = False 
    while (l <= r and x >= array[l]) and (x <= array[r] and not res):
        mid = l + int (((x - array[l]) / (array[r] - array[l])) * (r - l))
        b = array[mid]
        if (b == x):
            res = True 
            return res 
        if (b < x):
            l = mid + 1
        else:
            r = mid - 1 
    return res

#Exponential Search
def ExpBinarySeaarch(array, x, left, right):
    res = False
    while (left <= right and not res):
        mid = (left + right) // 2
        b = array[mid]
        if x == b:
            res = True 
            return res 
        if (b > x):
            right = mid - 1
        else: 
            left = mid + 1
    return res

def ExponentialSearch(array, x):
    bord = 1
    while (bord < len(array) - 1 and array[bord] < x):
        bord = bord * 2
    if (bord > len(array) - 1):
        bord = len(array) - 1
    return ExpBinarySeaarch(array, x, bord // 2, bord)
    
#Fibonacci Search
def FibonacciSearch(arr, target):
    
    n = len(arr)
    if n == 0:
        return -1

    fib_m_minus_2 = 0
    fib_m_minus_1 = 1
    fib_m = fib_m_minus_1 + fib_m_minus_2
    while fib_m < n:
        fib_m_minus_2 = fib_m_minus_1
        fib_m_minus_1 = fib_m
        fib_m = fib_m_minus_1 + fib_m_minus_2

    offset = -1

    while fib_m > 1:
        i = min(offset + fib_m_minus_2, n - 1)  

        if arr[i] < target:
            fib_m = fib_m_minus_1
            fib_m_minus_1 = fib_m_minus_2
            fib_m_minus_2 = fib_m - fib_m_minus_1
            offset = i
        elif arr[i] > target:
            fib_m = fib_m_minus_2
            fib_m_minus_1 = fib_m_minus_1 - fib_m_minus_2
            fib_m_minus_2 = fib_m - fib_m_minus_1
        else:
            return True

    if fib_m_minus_1 and arr[offset + 1] == target:
        return True

    return False  


# #Average Case
# x2_point = []
# y2_point = []
# n = 500
# while (n <= 10000):
#     A = [random.randint(0, n) for _ in range(n)]
#     A.sort()
#     x = A[random.randint(0, len(A) - 1)]
#     t = timeit.timeit(lambda: LinearSearch(A, x), number = n) / n
#     x2_point.append(n)
#     y2_point.append(t)
#     n = n + 500
#     print(2)
# #Worst Case
# x3_point = []
# y3_point = []
# n = 500
# while (n <= 10000):
#     A = [random.randint(0, n) for _ in range(n)]
#     A.sort()
#     x = A[-1]
#     t = timeit.timeit(lambda: LinearSearch(A, x), number = n) / n
#     x3_point.append(n)
#     y3_point.append(t)
#     n = n + 500
#     print(3)
# #Best Case
# x1_point = []
# y1_point = []
# n = 500
# while (n <= 10000):
#     A = [random.randint(0, n) for _ in range(n)]
#     A.sort()
#     x = A[0]
#     t = timeit.timeit(lambda: LinearSearch(A, x), number = n) / n
#     x1_point.append(n)
#     y1_point.append(t)
#     n = n + 500
#     print(1)
# Инвертирование списков
# x1_point.reverse()
# y1_point.reverse()
# x2_point.reverse()
# y2_point.reverse()
# x3_point.reverse()
# y3_point.reverse()

# x1_point = np.array(x1_point)
# y1_point = np.array(y1_point)
# x2_point = np.array(x2_point)
# y2_point = np.array(y2_point)
# x3_point = np.array(x3_point)
# y3_point = np.array(y3_point)

# # Константная регрессия для Best Case
# mean_y1 = np.mean(y1_point)
# x1_plot = np.linspace(min(x1_point), max(x1_point), 400)

# # Регрессия на основе sqrt(x) для Average Case
# model2 = LinearRegression()
# model2.fit(x2_point.reshape(-1, 1), y2_point)
# x2_plot = np.linspace(min(x2_point), max(x2_point), 400).reshape(-1, 1)
# # Регрессия на основе sqrt(x) для Worst Case
# model3 = LinearRegression()
# model3.fit(x3_point.reshape(-1, 1), y3_point)
# x3_plot = np.linspace(min(x3_point), max(x3_point), 400).reshape(-1, 1)

# # Построение графика
# plt.xlabel('array size')
# plt.ylabel('search time')
# plt.title("Linear Search")

# # Построение точек
# plt.plot(x1_point, y1_point, ".")
# plt.plot(x2_point, y2_point, ".")
# plt.plot(x3_point, y3_point, ".")

# # Построение линий регрессии
# plt.plot(x1_plot, np.full_like(x1_plot, mean_y1), color='green', label="Best Case O(1)")
# plt.plot(x2_plot, model2.predict(x2_plot), color='blue', label="Average Case O(n)")
# plt.plot(x3_plot, model3.predict(x3_plot), color='red', label="Worst Case O(n)")

# plt.legend()
# plt.grid(True)
# plt.show()
# x1_point = np.array(x1_point)
# y1_point = np.array(y1_point)
# x2_point = np.array(x2_point)
# y2_point = np.array(y2_point)
# x3_point = np.array(x3_point)
# y3_point = np.array(y3_point)

# # Константная регрессия для Best Case
# mean_y1 = np.mean(y1_point)
# x1_plot = np.linspace(min(x1_point), max(x1_point), 400)

# # Регрессия на основе sqrt(x) для Average Case
# x2_sqrt = np.sqrt(x2_point)
# model2 = LinearRegression()
# model2.fit(x2_sqrt.reshape(-1, 1), y2_point)
# x2_plot = np.linspace(min(x2_point), max(x2_point), 400).reshape(-1, 1)
# x2_plot_sqrt = np.sqrt(x2_plot)

# # Регрессия на основе sqrt(x) для Worst Case
# x3_sqrt = np.sqrt(x3_point)
# model3 = LinearRegression()
# model3.fit(x3_sqrt.reshape(-1, 1), y3_point)
# x3_plot = np.linspace(min(x3_point), max(x3_point), 400).reshape(-1, 1)
# x3_plot_sqrt = np.sqrt(x3_plot)

# # Построение графика
# plt.xlabel('array size')
# plt.ylabel('search time')
# plt.title("Jump Search")

# # Построение точек
# plt.plot(x1_point, y1_point, ".")
# plt.plot(x2_point, y2_point, ".")
# plt.plot(x3_point, y3_point, ".")

# # Построение линий регрессии
# plt.plot(x1_plot, np.full_like(x1_plot, mean_y1), color='green', label="Best Case O(1)")
# plt.plot(x2_plot, model2.predict(x2_plot_sqrt), color='blue', label="Average Case O(sqrt(n))")
# plt.plot(x3_plot, model3.predict(x3_plot_sqrt), color='red', label="Worst Case O(sqrt(n))")

# plt.legend()
# plt.grid(True)
# plt.show()




# # Using Numpy to create an array X
# x = np.arange(1, 100)

# # Assign variables to the y axis part of the curve
# y1 = np.log(x)
# y2 = np.log(np.log(x))
# y3 = np.sqrt(x)
# y4 = x

# # Plotting both the curves simultaneously
# plt.plot(x, y1, color='r', label='O(log(n))')
# plt.plot(x, y2, color='g', label='O(log(log(n)))')
# plt.plot(x, y3, color='b', label='O(sqrt(n))')
# plt.plot(x, y4, color='purple', label='O(n)')

# plt.grid(True)
# # Naming the x-axis, y-axis and the whole graph
# plt.xlabel("array size")
# plt.ylabel("T(n)")
# plt.title("Comparison")

# # Adding legend, which helps us recognize the curve according to it's color
# plt.legend()

# # To load the display window
# plt.show()



# q = [1, 2, 3, 5, 6, 8, 12 ,45, 78, 98]
# if FibonacciSearch(q, 0):
#     print("YES")
# else:
#     print("NO")
    


