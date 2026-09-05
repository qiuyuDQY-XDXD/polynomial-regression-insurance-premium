import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

# 数据文件与脚本在同一目录；用 __file__ 定位，避免在其他工作目录下运行时找不到文件
data_path = Path(__file__).parent / '中国人寿.xlsx'
data = pd.read_excel(data_path)
# print('原始数据\n',data.head())
# 绘制保费分布的核密度估计图
# 什么样的特征是好特征
# sns.kdeplot(data=data, x='charges', hue='sex', fill=True)
# sns.kdeplot(data=data, x='charges', hue='smoker', fill=True)
# sns.kdeplot(data=data, x='charges', hue='region', fill=True)
# 区域，性别对保费影响不大
# sns.kdeplot(data=data, x='charges', hue='children', fill=True)
# plt.show()
# 特征工程，删除不重要的特征
data = data.drop(columns=['region','sex'])
# print('删除不重要的特征后\n',data.head())
def convert(df,bmi):
    df['bmi'] = 'fat' if df['bmi'] >= bmi else 'standard'
    return df
data = data.apply(convert,axis=1,args=(30,))
# print('转换bmi后的数据\n',data.head())
data = pd.get_dummies(data)
# print('one-hot编码后的数据\n',data.head())

# 数据提取
X = data.drop(columns=['charges'])
y = data['charges']
# print('原始数据\n',data.head())
# print('特征X\n',X.head())
# print('目标y\n',y.head())

# 数据升维
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
# poly = PolynomialFeatures(degree=2,include_bias=True)
X_poly = poly.fit_transform(X)
# print('升维后的特征X\n',X_poly)
# print('目标y\n',y)

# 数据拆分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=1024)

# 线性回归+评估
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
model = LinearRegression()
model.fit(X_train,y_train)
# 评估R^2
print('训练集上的R^2值为',model.score(X_train,y_train))
print('预测集上的R^2值为',model.score(X_test,y_test))

# 均方误差
import numpy as np
print('训练集上的均方误差为',np.sqrt(mean_squared_error(y_train,model.predict(X_train))))
print('预测集上的均方误差为',np.sqrt(mean_squared_error(y_test,model.predict(X_test))))

print('训练数据对数误差为',np.sqrt(mean_squared_log_error(y_train,model.predict(X_train))))
print('预测数据对数误差为',np.sqrt(mean_squared_log_error(y_test,model.predict(X_test))))
