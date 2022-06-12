# ================基于XGBoost原生接口的分类=============
import numpy as np
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 准确率
# 加载样本数据集
# iris = load_iris()
# X,y = iris.data,iris.target
# X,y = iris.data,iris.target

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 10,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()
print(plst)


dtrain = xgb.DMatrix(feature, label) # 生成数据集格式
num_rounds = 500
print(dtrain)
model = xgb.train(params, dtrain, num_rounds) # xgboost模型训练
