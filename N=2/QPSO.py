import numpy as np
from pyswarm import pso
import pandas as pd
from scipy.special import factorial
from scipy.special import gamma

def NIPDGM1Mlandai(X1, X2, X3, order1, order2, order3,  lamda1,  lamda2, lamda3):

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

    x1_ago = NFAGO(X1, order1, lamda1)
    x2_ago = NFAGO(X2, order2, lamda2)
    x3_ago = NFAGO(X3, order3, lamda3)
    xi = np.array([x1_ago, x2_ago, x3_ago])

    # 3.紧邻均值生成序列
    m = len(x1_ago)

    # 3.紧邻均值生成序列
    def JingLing(m):
        return [(m[j] + m[j - 1]) / 2 for j in range(1, len(m))]

    Z1 = JingLing(x1_ago)
    Z2 = JingLing(x2_ago)
    Z3 = JingLing(x3_ago)
    Zi = np.array([Z1, Z2, Z3])
    Zii = np.array([Z2, Z3])

    # 4.求相关参数
    # 4.1 求Y矩阵
    Y = []
    for i in range(1, len(x1_ago)):
        y = x1_ago[i] - x1_ago[i - 1]
        Y.append(y)
    Y = np.array(Y).reshape(-1, 1)

    # 4.1 B矩阵
    B = []
    B1 = []
    B1 = np.transpose(Zi)

    B2 = np.ones((m - 1, 1), dtype=int)

    B3 = []

    def generate_matrix(m, N):
        matrix = np.zeros((m - 1, N))
        # 填充矩阵
        for i in range(2, m + 1):  # 从2到m-1（行）
            for j in range(2, N + 2):  # 从1到N（列）
                matrix[i - 2, j - 2] = (i ** j - (i - 1) ** j) / j

        return matrix

    B3 = generate_matrix(m, 2)

    # 拼接
    B = np.hstack((B1, B2, B3))

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

    a1 = p[0]
    a2 = p[1]
    a3 = p[2]
    β0 = p[3]
    β1 = p[4]
    β2 = p[5]
    aa1 = np.array([a1, a2, a3])
    aa2 = np.array([a2, a3])

    # 6.生成拟合模型
    # 6.1计算参数
    μ1 = 0.5 * a1 / (1 - 0.5 * a1)
    μ2 = 1 / (1 - 0.5 * a1)
    μ3 = β0 / (1 - 0.5 * a1)
    μ4 = β1 / (1 - 0.5 * a1)
    μ5 = β2 / (1 - 0.5 * a1)

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

    Y = YNFAGO(T, order1, lamda1)

    return np.array(Y)

def MAPE(Y_true, Y_pred):
    return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

def objective(params, X1, X2, X3):
    # 目标函数，计算给定参数下模型的MAPE
    order1, order2, order3,  lamda1,  lamda2, lamda3 = params
    Y = NIPDGM1Mlandai(X1, X2, X3, order1, order2, order3,  lamda1,  lamda2, lamda3)
    return MAPE(X1, Y)

def optimize_parameters(X1, X2, X3):
    bounds = [(0.001, 1.05)] * 3 + [(0.001, 1.0)] * 3

    def pso_obj(params):

        fx = objective(params, X1, X2, X3)
        print(f"Params: {params}, MAPE: {fx}")
        return fx

    result = pso(pso_obj, [b[0] for b in bounds], [b[1] for b in bounds], swarmsize=100, maxiter=200)

    if len(result) == 6:
        order1_opt, order2_opt, order3_opt, lamda1_opt, lamda2_opt, lamda3_opt = result
    else:
        print("Error: Unexpected length of result from pso function")
        return None, None, None, None

    return order1_opt, order2_opt, order3_opt, lamda1_opt, lamda2_opt, lamda3_opt


if __name__ == "__main__":
    df5 = pd.read_excel('D:/研究生/专业学习/灰色预测/要写/锂电池/test/NASAtest.xlsx', sheet_name='B51')
    X1 = np.array(df5.iloc[:19, 0].tolist())
    X2 = np.array(df5.iloc[:19, 1].tolist())
    X3 = np.array(df5.iloc[:19, 2].tolist())
    order1_opt, order2_opt, order3_opt, lamda1_opt, lamda2_opt, lamda3_opt = optimize_parameters(X1, X2, X3)

    Y = NIPDGM1Mlandai(X1, X2, X3, order1_opt, order2_opt, order3_opt, lamda1_opt, lamda2_opt, lamda3_opt)
    error = MAPE(X1, Y)

    print(f"优化后的 alpha: {order1_opt}")
    print(f"优化后的 alpha: {order2_opt}")
    print(f"优化后的 alpha: {order3_opt}")
    print(f"优化后的 landa1: {lamda1_opt}")
    print(f"优化后的 landa2: {lamda2_opt}")
    print(f"优化后的 landa3: {lamda3_opt}")
    print(f"预测序列: {Y}")
    print(f"MAPE: {error}%")
else:
    print("优化失败")