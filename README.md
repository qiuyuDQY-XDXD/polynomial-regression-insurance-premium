# 多项式回归预测中国人寿保费

基于中国人寿保险数据集，使用多项式回归（Polynomial Features + Linear Regression）预测客户保费金额。

## 项目流程

1. **数据加载**：读取中国人寿保险数据集（Excel）
2. **特征选择**：通过 KDE 核密度估计图可视化分析，删除对保费影响较小的特征（region、sex）
3. **特征工程**：
   - BMI 离散化：将连续 BMI 值转为 `fat` / `standard` 分类
   - One-Hot 编码：将分类变量转为 0/1 数值特征
4. **多项式升维**：使用 `PolynomialFeatures(degree=2, interaction_only=True)` 生成特征交叉项，捕捉特征间的交互效应
5. **模型训练**：使用 `LinearRegression`（正规方程）拟合训练数据
6. **模型评估**：通过 R²、RMSE、对数均方误差（MSLE）评估训练集和测试集表现

## 技术栈

- Python
- pandas（数据处理）
- scikit-learn（多项式特征、线性回归、模型评估）
- seaborn / matplotlib（数据可视化）

## 数据说明

| 字段 | 说明 |
|------|------|
| age | 年龄 |
| sex | 性别（特征选择后删除） |
| bmi | 体质指数（离散化为 fat/standard） |
| children | 子女数量 |
| smoker | 是否吸烟 |
| region | 地区（特征选择后删除） |
| charges | 保费（预测目标） |
