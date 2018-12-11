import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

examDict = {'学习时间': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75,
                     2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
            '分数': [10, 22, 13, 43, 20, 22, 33, 50, 62,
                   48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}
examDf = DataFrame(examDict)
'''
# 绘制散点图
plt.scatter(examDf.分数, examDf.学习时间, color='b', label="Exam Data")

# 添加图的标签（x轴，y轴）
plt.xlabel("Hours")
plt.ylabel("Score")
# 显示图像
plt.show()

rDf = examDf.corr()
print(rDf)
'''

X_train, X_test, Y_train, Y_test = train_test_split(
    examDf.学习时间, examDf.分数, train_size=.8)
'''
print("原始数据特征:", examDf.学习时间.shape,
      ",训练数据特征:", X_train.shape,
      ",测试数据特征:", X_test.shape)

print("原始数据标签:", examDf.分数.shape,
      ",训练数据标签:", Y_train.shape,
      ",测试数据标签:", Y_test.shape)
'''
'''
# 散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
plt.scatter(X_test, Y_test, color="red", label="test data")
# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Pass")
# 显示图像
plt.show()
'''
model = LinearRegression()

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
Y_train = Y_train.values.reshape(-1, 1)
Y_test = Y_test.values.reshape(-1, 1)

model.fit(X_train, Y_train)
a = model.intercept_  # 截距

b = model.coef_  # 回归系数

print("最佳拟合线:截距", a, ",回归系数：", b)

# 训练数据的预测值
y_train_pred = model.predict(X_train)
# 绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='black', linewidth=3, label="best line")

# 训练数据散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
# 测试数据散点图
plt.scatter(X_test, Y_test, color='red', label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
# 显示图像
# plt.savefig("lines.jpg")
score = model.score(X_test, Y_test)
print("决定系数R: " + str(score))
plt.show()
