# 函数拟合报告

## 1. 函数定义

本实验选择拟合如下目标函数：

```math
f(x) = \sin(2x) + 0.3x^2
```

该函数同时包含非线性振荡项和二次项，适合用于验证基于 ReLU 的全连接网络是否具备较好的函数逼近能力。

## 2. 数据采集

- 自变量采样区间：`[-3, 3]`
- 训练集：在区间内等间隔采样 `512` 个点
- 测试集：在区间内等间隔采样 `256` 个点
- 标签值：直接由目标函数 `f(x)` 计算得到

训练集用于参数学习，测试集用于评估网络在未参与训练样本上的拟合误差。

## 3. 模型描述

模型采用 NumPy 手写实现的三层全连接网络：

- 输入层维度：`1`
- 隐藏层 1：`64` 个神经元，激活函数为 `ReLU`
- 隐藏层 2：`64` 个神经元，激活函数为 `ReLU`
- 输出层：`1` 个神经元，直接输出拟合值

训练配置如下：

- 优化方式：小批量梯度下降
- 学习率：`0.003`
- 损失函数：均方误差 `MSE`
- 训练轮数：`4000`
- batch size：`64`

## 4. 拟合效果

训练完成后，脚本会输出测试集上的 `MSE` 和 `MAE`，并保存：

- `function_fitting_predictions.csv`：测试集上的 `x`、真实值、预测值
- `function_fitting_result.svg`：左侧是真实函数和预测函数，右侧是训练损失曲线

从实验结果可以观察到：

- 预测曲线能够较好贴合目标函数整体趋势
- 训练误差会随训练过程明显下降
- 两层 ReLU 隐藏层的全连接网络能够有效逼近该一维非线性函数

## 5. 代码文件

- 训练脚本：[function_fitting_relu.py](/Users/ethan/Documents/GitHub/exercise/chap4_ simple neural network/function_fitting_relu.py)
- 报告文件：[function_fitting_report.md](/Users/ethan/Documents/GitHub/exercise/chap4_ simple neural network/function_fitting_report.md)
