import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import factorial
import pandas as pd
from scipy.linalg import expm

df = pd.read_excel('D:/研究生/专业学习/灰色预测/要写/锂电池/NASAlidianchi/NASA.xlsx', sheet_name='B7归一化')
# 1.这里将 x1 作为特征序列 x2,x3,x4,x5作为相关因素序列。19个拟合，5个预测
X1 = df.iloc[:134, 2].tolist()
X2 = df.iloc[:134, 0].tolist()
X3 = df.iloc[:134, 1].tolist()
#测试预测值数据 自变量序列
XC2 = df.iloc[134:167, 0].tolist()
XC3 = df.iloc[134:167, 1].tolist()
#B7
orders = [1.00035599, 0.97460081, 0.64482443]
lamda = [0.96085017, 0.35159757, 0.03272047]

#预测的个数和真实值
h = 33
forezsz = df.iloc[134:167, 2].tolist()
allX1 = df.iloc[:, 2].tolist()
N = 2

# 2.累加序列，生成新序列
def NFAGO(X0, order, lamda):
    m = len(X0)
    F = np.zeros(m)
    # 计算每个时间步的累加值
    for k in range(m):  # 0到m-1
        sum_val = 0
        for i in range(k + 1):  # 0到k
            # 计算权重 cl[v, i]
            numerator = factorial(order + k - i)
            denominator = factorial(k - i + 1) * factorial(order)
            cl_vi = (numerator / denominator) * (lamda ** (k - i))
            sum_val += cl_vi * X0[i]
        F[k] = sum_val
    return F

x1_ago = NFAGO(X1, orders[0], lamda[0])
x2_ago = NFAGO(X2, orders[1], lamda[1])
x3_ago = NFAGO(X3, orders[2], lamda[2])
xi = np.array([x1_ago, x2_ago, x3_ago])
print("xi:", xi)

m = len(x1_ago)
# 3.紧邻均值生成序列
def JingLing(m):
    return [(m[j] + m[j - 1]) / 2 for j in range(1, len(m))]

Z1 = JingLing(x1_ago)
Z2 = JingLing(x2_ago)
Z3 = JingLing(x3_ago)
Zi = np.array([Z1, Z2, Z3])
Zii = np.array([Z2, Z3])
print("Zi.shape:", Zi.shape)
print("Zi", Zi)
print("Zii", Zii)

# 4. 计算相关参数
# 4.1 Y矩阵
Y = []
for i in range(1, len(x1_ago)):
    y = x1_ago[i] - x1_ago[i-1]
    Y.append(y)
Y = np.array(Y).reshape(-1, 1)
print("Y矩阵:")
print(Y)
print("Y.shape:", Y.shape)

# 4.1 B矩阵
B = []

B1 = []
B1 = np.transpose(Zi)
print("B1.shape:", B1.shape)
print("B1:", B1)

B2 = np.ones((m-1, 1), dtype=int)
print("B2:", B2)

B3 = []
def generate_matrix(m, N):
    matrix = np.zeros((m-1, N))
    # 填充矩阵
    for i in range(2, m + 1):  # 从2到m-1（行）
        for j in range(2, N + 2):  # 从1到N（列）
            matrix[i - 2, j - 2] = (i ** j - (i - 1) ** j) / j

    return matrix
B3 = generate_matrix(m, N)
print("B3:", B3)

# 拼接
B = np.hstack((B1, B2, B3))
print("B:", B)
print("B.shape:", B.shape)

# 5 由第四步求出参数
# 用最小二乘参数估计
m = len(B) + 1
def JISUAN(B, Y):
    m = len(B) + 1
    # 计算矩阵的行列式
    det = np.linalg.det

    if m == 3 + 3 and det(B) != 0:
        p = np.dot(np.linalg.inv(B), Y)
    elif m > 3 + 3 and det(np.dot(B.T, B)) != 0:
        p = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    elif m < 3 + 3 and det(np.dot(B, B.T)) != 0:
        p = B.T.dot(np.linalg.inv(B.dot(B.T))).dot(Y)
    else:
        p = None
    return p
p = JISUAN(B, Y)
print("p =", p)

a1 = p[0]
print("a1:", a1)
a2 = p[1]
print("a2:", a2)
a3 = p[2]
print("a3:", a3)
β0 = p[3]
print("β0:", β0)
β1 = p[4]
print("β1:", β1)
β2 = p[5]
print("β2:", β2)
aa1 = np.array([a1, a2, a3])
aa2 = np.array([a2, a3])
print("aa1:", aa1)
print("aa2:", aa2)

# 6.生成拟合模型
# 6.1计算参数
μ1 = 0.5 * a1 / (1 - 0.5 * a1)
μ2 = 1 / (1 - 0.5 * a1)
μ3 = β0 / (1 - 0.5 * a1)
μ4 = β1 / (1 - 0.5 * a1)
μ5 = β2 / (1 - 0.5 * a1)
print(μ1, μ2, μ3, μ4, μ5)


# 拟合部分 (2到len(x1_ago))
T = [x1_ago[0]]  # 初始值
for k in range(2, len(x1_ago) + 1):
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0

    # 计算s1，仅使用x1_ago数据
    for u in range(2, k + 1):
        for i in range(2, 3 + 1):
            s1 += μ2 * aa2[i - 2] * Zii[i - 2][u - 2]  # Zii是基于x1_ago计算的

    # 计算s2
    s2 += μ3 * (k - 1) + ((k ** 2 - 1) / 2) * μ4 + ((k ** 3 - 1) / 3) * μ5

    # 计算s3
    s3 = μ2 * x1_ago[0] + μ1 * x1_ago[k - 2]

    # 计算s4
    for u in range(2, k):  # 2到k-1
        s4 += μ2 * a1 * Z1[u - 2]  # Z1是基于x1_ago计算的

    T.append(s1 + s2 + s3 + s4)

T = [float(x) for x in T]
print("T:", T)

#6.2 新息分数阶优先累减
def YNFAGO(T0, order, lamda):
    m = len(T0)
    Y = np.zeros(m)  # 初始化累减还原序列 T
    for k in range(1, m + 1):  # 从 1 到 m
        jian_val = 0
        for i in range(1, k + 1):  # 1 到 k
            # 计算 numerator注意范围
            numerator = 1  # 初始化为 1
            for p in range(0, k - i):  # 从 1 到 k-i
                numerator *= (-order + p)  # 逐步累乘
            denominator = gamma(k - i + 1)
            # 计算权重 cl_vi
            cl_vi = (numerator / denominator) * (lamda ** (k - i))
            # 累加权重乘以 T0 的值
            jian_val += cl_vi * T0[i - 1]  # 使用 +=
        Y[k - 1] = jian_val  # 赋值到 Y[k-1]
    return Y

Y = YNFAGO(T, orders[0], lamda[0])
print("拟合序列：", Y)

# 计算 APE（Absolute Percentage Error）
ape = np.abs(np.subtract(Y[1:], X1[1:])) / np.maximum(np.abs(X1[1:]), np.finfo(np.float64).eps)
# 使用 np.finfo(np.float64).eps 避免除以零的情况

# 计算 MAPE（Mean Absolute Percentage Error）
mape = np.mean(ape) * 100

# 8.3 计算 RMSE（Root Mean Squared Error）
rmse = np.sqrt(np.mean((np.subtract(Y[1:], X1[1:])) ** 2))

print("拟合RMSE:", rmse)
print("拟合MAPE:", mape, "%")

#7. 预测部分：增加X2、X3，并计算Zcii和Zc1
X2.extend(XC2)
X3.extend(XC3)
print("X3:", X3)
xc2_ago = NFAGO(X2, orders[1], lamda[1])
print("xc2_ago:", xc2_ago)
xc3_ago = NFAGO(X3, orders[2], lamda[2])
Zc2 = JingLing(xc2_ago)
Zc3 = JingLing(xc3_ago)
Zcii = np.array([Zc2, Zc3])
#print("Zcii:", Zcii)

# 初始值设定
xc1_ago = x1_ago.copy()
Zc1 = JingLing(xc1_ago)
print("Zc1初始值:", Zc1)
T_pred = []  # 初始预测值

# 7.2 预测：从len(x1_ago)到len(x1_ago) + h进行预测
for k in range(len(x1_ago) + 1, len(x1_ago) + h + 1):  # 从len(x1_ago)到len(x1_ago)+h
    s1, s2, s3, s4 = 0, 0, 0, 0

    # 计算s1
    for u in range(2, k + 1):
        for i in range(2, 3 + 1):
            if u - 2 < len(Zcii[i - 2]):
                s1 += μ2 * aa2[i - 2] * Zcii[i - 2][u - 2]

    # 计算s2
    s2 = μ3 * (k - 1) + ((k ** 2 - 1) / 2) * μ4 + ((k ** 3 - 1) / 3) * μ5

    # 计算s3
    s3 = μ2 * x1_ago[0] + μ1 * xc1_ago[k - 2]

    # 计算s4
    for u in range(2, k):  # 2到k-1
        if u - 2 < len(Zc1):
            s4 += μ2 * a1 * Zc1[u - 2]

    # 当前预测值
    p = s1 + s2 + s3 + s4
    T_pred.append(p)

    # 更新历史数据：将当前预测值 p 添加到 xc1_ago 的末尾
    xc1_ago = np.append(xc1_ago, p)  # 使用 np.append() 更新 xc1_ago

    # 使用更新后的 xc1_ago 计算新的 Zc1
    Zc1 = JingLing(xc1_ago)

    # 输出当前的 xc1_ago 和 Zc1
    #print(f"迭代{k + 1}时的 xc1_ago:", xc1_ago)
    #print(f"迭代{k + 1}时的 Zc1:", Zc1)

T_pred = [float(x) for x in T_pred]
print("T_pred:", T_pred)
T.extend(T_pred)
print(T)

Y = YNFAGO(T, orders[0], lamda[0])
print("全部值：", Y)
fore0 = Y[-h:]
print("预测序列:", fore0)

# 计算 APE（Absolute Percentage Error）
foreape = np.abs(np.subtract(fore0, forezsz)) / np.maximum(np.abs(forezsz), np.finfo(np.float64).eps)
# 使用 np.finfo(np.float64).eps 避免除以零的情况
# 计算 MAPE（Mean Absolute Percentage Error）
foremape = np.mean(foreape) * 100
# 8.3 计算 RMSE（Root Mean Squared Error）
yucermse = np.sqrt(np.mean((np.subtract(fore0,  forezsz)) ** 2))
print("预测RMSE:", yucermse)
print("预测MAPE:", foremape, "%")

# 8. 评价指标
# 8.1 计算 APE（Absolute Percentage Error）
allAPE = np.abs(np.subtract(Y[1:], allX1[1:])) / np.maximum(np.abs(allX1[1:]), np.finfo(np.float64).eps)
# 使用 np.finfo(np.float64).eps 避免除以零的情况
# 8.2 计算 MAPE（Mean Absolute Percentage Error）
allMAPE = np.mean(allAPE) * 100
print("整体MAPE:", allMAPE, "%")
# 8.3 计算 RMSE（Root Mean Squared Error）
rmse = np.sqrt(np.mean((np.subtract(Y[1:], allX1[1:])) ** 2))
print("RMSE:", rmse)
# 8.4 计算 APE 的 STD
ape_std = np.std(allAPE)
print("STD:", ape_std)
# 8.4 计算 APE 的最大值
ape_max = np.max(allAPE)
ape_maxbaifen = ape_max * 100
print("APE最大值:", ape_maxbaifen, "%")

r = range(len(allX1))
t = list(r)
plt.plot(t, allX1,color='r',linestyle="--",label='true')
plt.plot(t,Y,color='b',linestyle="--",label="predict")
plt.legend(loc='upper right')
plt.show()

# 将 Y 转换为 DataFrame（如果 Y 是二维的，或者将其包装为单列数据）
df = pd.DataFrame(Y)

# 将 DataFrame 保存为 Excel 文件
df.to_excel('output.xlsx', index=False)

print("Y 已保存为 Excel 文件。")