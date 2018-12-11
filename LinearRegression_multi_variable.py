import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#通过read_csv来读取我们的目的数据集
adv_data = pd.read_csv("./Advertising.csv")
#清洗不需要的数据
new_adv_data = adv_data.ix[:,1:]
#得到我们所需要的数据集且查看其前几列以及数据形状
print('head:',new_adv_data.head(),'\nShape:',new_adv_data.shape)

#数据描述
print(new_adv_data.describe())
#缺失值检验
print(new_adv_data[new_adv_data.isnull()==True].count())

new_adv_data.boxplot()
plt.show()
##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
#相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
print(new_adv_data.corr())