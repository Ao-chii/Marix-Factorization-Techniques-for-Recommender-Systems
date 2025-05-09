import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.linalg import solve

class MF():
    def __init__(self, R, K, alpha, beta, iterations):
        """
        执行矩阵分解以预测矩阵中的空值。

        参数
        - R (ndarray)   : 用户-项目评分矩阵
        - K (int)       : 潜在维度的数量
        - alpha (float) : 学习率
        - beta (float)  : 正则化参数
        """

        # 初始化用户-项目评分矩阵
        self.R = R
        # 获取用户数量和项目数量
        self.num_users, self.num_items = R.shape
        # 设置潜在维度的数量
        self.K = K
        # 设置学习率
        self.alpha = alpha
        # 设置正则化参数
        self.beta = beta
        # 设置迭代次数
        self.iterations = iterations

    def train_sgd(self):
        # 初始化用户和项目的潜在特征矩阵
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # 初始化偏置项
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # 创建训练样本列表
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # 执行随机梯度下降，迭代次数为 self.iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # 每 100 次迭代打印一次误差
            if (i+1) % 100 == 0:
                print("SGD Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        计算总均方误差的函数
        """
        xs, ys = self.R.nonzero()
        # 获取非零元素的索引
        predicted = self.full_matrix()
        # 生成预测矩阵
        error = 0
        # 初始化误差为0
        for x, y in zip(xs, ys):
            # 遍历非零元素的索引
            error += pow(self.R[x, y] - predicted[x, y], 2)
            # 计算实际值与预测值之间的平方差，并累加到误差中
        return np.sqrt(error)
        # 返回均方根误差

    def sgd(self):
        """
        执行随机梯度下降
        """
        for i, j, r in self.samples:
            # 计算预测值和误差
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # 更新用户和项目的偏置
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # 更新用户和项目的潜在特征矩阵
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        获取用户 i 对项目 j 的预测评分

        参数:
        i (int): 用户的索引
        j (int): 项目的索引

        返回:
        float: 预测评分
        """
        # 计算预测评分，包括全局偏差、用户偏差、项目偏差和用户-项目交互项
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        计算使用结果偏差、P 和 Q 生成的完整矩阵
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

    def train_als(self):
        """
        使用交替最小二乘法(ALS)训练模型
        """
        # 初始化用户和物品的潜在特征矩阵
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # 初始化偏置项
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # 创建评分矩阵的掩码(mask)，标记哪些位置有评分
        mask = self.R > 0

        # 执行交替最小二乘法，迭代次数为 self.iterations
        training_process = []
        for i in range(self.iterations):
            # 固定Q，更新P
            self.update_user_features(mask)

            # 固定P，更新Q
            self.update_item_features(mask)

            # 更新偏置项
            self.update_biases(mask)

            # 计算误差
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 100 == 0:
                print("ALS Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def update_user_features(self, mask):
        """更新用户特征矩阵P"""
        for u in range(self.num_users):
            # 获取用户u评分过的物品索引
            items_u = np.where(mask[u])[0]

            # 如果用户没有评分，跳过
            if len(items_u) == 0:
                continue

            # 获取物品特征矩阵的子集
            Q_u = self.Q[items_u]

            # 获取评分向量
            ratings_u = self.R[u, items_u]

            # 减去偏置项的影响
            ratings_u_adj = ratings_u - self.b - self.b_u[u] - self.b_i[items_u]

            # 构建线性方程组 Q_u^T * Q_u * P_u = Q_u^T * ratings_u_adj
            A = Q_u.T.dot(Q_u) + self.beta * np.eye(self.K)
            b = Q_u.T.dot(ratings_u_adj)

            # 求解线性方程组
            self.P[u] = solve(A, b)

    def update_item_features(self, mask):
        """更新物品特征矩阵Q"""
        for i in range(self.num_items):
            # 获取评分过物品i的用户索引
            users_i = np.where(mask[:, i])[0]

            # 如果物品没有评分，跳过
            if len(users_i) == 0:
                continue

            # 获取用户特征矩阵的子集
            P_i = self.P[users_i]

            # 获取评分向量
            ratings_i = self.R[users_i, i]

            # 减去偏置项的影响
            ratings_i_adj = ratings_i - self.b - self.b_u[users_i] - self.b_i[i]

            # 构建线性方程组 P_i^T * P_i * Q_i = P_i^T * ratings_i_adj
            A = P_i.T.dot(P_i) + self.beta * np.eye(self.K)
            b = P_i.T.dot(ratings_i_adj)

            # 求解线性方程组
            self.Q[i] = solve(A, b)

    def update_biases(self, mask):
        """更新偏置项"""
        # 更新用户偏置
        for u in range(self.num_users):
            items_u = np.where(mask[u])[0]
            if len(items_u) > 0:
                error = self.R[u, items_u] - (self.b + self.b_i[items_u] + self.P[u].dot(self.Q[items_u].T))
                self.b_u[u] = np.sum(error) / (len(items_u) + self.beta)

        # 更新物品偏置
        for i in range(self.num_items):
            users_i = np.where(mask[:, i])[0]
            if len(users_i) > 0:
                predictions = self.b + self.b_u[users_i] + np.sum(self.P[users_i] * self.Q[i], axis=1)
                error = self.R[users_i, i] - predictions
                self.b_i[i] = np.sum(error) / (len(users_i) + self.beta)

    def compare_methods(self, test_ratio=0.2):
        """
        比较SGD和ALS两种优化方法的性能

        参数:
        test_ratio (float): 测试集比例

        返回:
        dict: 包含两种方法的训练过程和测试误差
        """
        # 创建训练集和测试集
        mask = self.R > 0
        test_mask = np.zeros_like(self.R, dtype=bool)

        # 随机选择一部分评分作为测试集
        for u in range(self.num_users):
            items_u = np.where(mask[u])[0]
            if len(items_u) > 0:
                n_test = max(1, int(len(items_u) * test_ratio))
                test_items = np.random.choice(items_u, n_test, replace=False)
                test_mask[u, test_items] = True

        # 创建训练集掩码
        train_mask = mask & ~test_mask

        # 创建训练集评分矩阵
        R_train = self.R.copy()
        R_train[test_mask] = 0

        # 保存原始评分矩阵
        R_original = self.R.copy()

        # ----- SGD训练 -----
        # 创建新的MF实例用于SGD
        mf_sgd = MF(R_train.copy(), K=self.K, alpha=self.alpha, beta=self.beta, iterations=self.iterations)

        # 修改train_sgd方法来记录训练集和测试集上的误差
        sgd_process = []

        # 初始化
        mf_sgd.P = np.random.normal(scale=1. / mf_sgd.K, size=(mf_sgd.num_users, mf_sgd.K))
        mf_sgd.Q = np.random.normal(scale=1. / mf_sgd.K, size=(mf_sgd.num_items, mf_sgd.K))
        mf_sgd.b_u = np.zeros(mf_sgd.num_users)
        mf_sgd.b_i = np.zeros(mf_sgd.num_items)
        mf_sgd.b = np.mean(mf_sgd.R[np.where(mf_sgd.R != 0)])

        # 创建训练样本
        mf_sgd.samples = [
            (i, j, mf_sgd.R[i, j])
            for i in range(mf_sgd.num_users)
            for j in range(mf_sgd.num_items)
            if mf_sgd.R[i, j] > 0
        ]

        # SGD训练循环
        for i in range(mf_sgd.iterations):
            np.random.shuffle(mf_sgd.samples)
            mf_sgd.sgd()

            # 计算训练集上的RMSE
            pred = mf_sgd.full_matrix()
            train_rmse = np.sqrt(np.mean((R_original[train_mask] - pred[train_mask]) ** 2))

            # 计算测试集上的RMSE
            test_rmse = np.sqrt(np.mean((R_original[test_mask] - pred[test_mask]) ** 2))

            sgd_process.append((i, train_rmse, test_rmse))

            # 每10次迭代打印一次
            if (i + 1) % 10 == 0:
                print(f"SGD Iteration: {i + 1}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # 最终的SGD预测
        sgd_pred = mf_sgd.full_matrix()
        sgd_final_rmse = np.sqrt(np.mean((R_original[test_mask] - sgd_pred[test_mask]) ** 2))

        # ----- ALS训练 -----
        # 创建新的MF实例用于ALS
        mf_als = MF(R_train.copy(), K=self.K, alpha=self.alpha, beta=self.beta, iterations=self.iterations)

        # ALS训练循环
        als_process = []

        # 初始化
        mf_als.P = np.random.normal(scale=1. / mf_als.K, size=(mf_als.num_users, mf_als.K))
        mf_als.Q = np.random.normal(scale=1. / mf_als.K, size=(mf_als.num_items, mf_als.K))
        mf_als.b_u = np.zeros(mf_als.num_users)
        mf_als.b_i = np.zeros(mf_als.num_items)
        mf_als.b = np.mean(mf_als.R[np.where(mf_als.R != 0)])

        # 创建评分矩阵的掩码
        als_mask = mf_als.R > 0

        # ALS训练循环
        for i in range(mf_als.iterations):
            # 固定Q，更新P
            mf_als.update_user_features(als_mask)

            # 固定P，更新Q
            mf_als.update_item_features(als_mask)

            # 更新偏置项
            mf_als.update_biases(als_mask)

            # 计算训练集上的RMSE
            pred = mf_als.full_matrix()
            train_rmse = np.sqrt(np.mean((R_original[train_mask] - pred[train_mask]) ** 2))

            # 计算测试集上的RMSE
            test_rmse = np.sqrt(np.mean((R_original[test_mask] - pred[test_mask]) ** 2))

            als_process.append((i, train_rmse, test_rmse))

            # 每10次迭代打印一次
            if (i + 1) % 10 == 0:
                print(f"ALS Iteration: {i + 1}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # 最终的ALS预测
        als_pred = mf_als.full_matrix()
        als_final_rmse = np.sqrt(np.mean((R_original[test_mask] - als_pred[test_mask]) ** 2))

        # 返回比较结果
        return {
            'sgd_process': sgd_process,
            'als_process': als_process,
            'sgd_test_rmse': sgd_final_rmse,
            'als_test_rmse': als_final_rmse,
            'test_ratio': test_ratio,
            'train_mask': train_mask,
            'test_mask': test_mask
        }

    def visualize_comparison(self, comparison_results):
        """
        可视化SGD和ALS方法的比较结果

        参数:
        comparison_results (dict): compare_methods方法的返回结果
        """
        # 提取训练过程数据
        sgd_iterations = [x[0] for x in comparison_results['sgd_process']]
        sgd_train_errors = [x[1] for x in comparison_results['sgd_process']]
        sgd_test_errors = [x[2] for x in comparison_results['sgd_process']]

        als_iterations = [x[0] for x in comparison_results['als_process']]
        als_train_errors = [x[1] for x in comparison_results['als_process']]
        als_test_errors = [x[2] for x in comparison_results['als_process']]

        # 创建图形
        plt.figure(figsize=(15, 10))

        # 1. 训练误差曲线对比
        plt.subplot(2, 2, 1)
        plt.plot(sgd_iterations, sgd_train_errors, 'b-', label='SGD - Training')
        plt.plot(als_iterations, als_train_errors, 'r-', label='ALS - Training')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Training Error')
        plt.legend()
        plt.grid(True)

        # 2. 测试误差曲线对比
        plt.subplot(2, 2, 2)
        plt.plot(sgd_iterations, sgd_test_errors, 'b--', label='SGD - Test')
        plt.plot(als_iterations, als_test_errors, 'r--', label='ALS - Test')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Test Error')
        plt.legend()
        plt.grid(True)

        # 3. SGD的训练集和测试集误差对比
        plt.subplot(2, 2, 3)
        plt.plot(sgd_iterations, sgd_train_errors, 'b-', label='Training')
        plt.plot(sgd_iterations, sgd_test_errors, 'b--', label='Test')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('SGD Method: Training vs Test')
        plt.legend()
        plt.grid(True)

        # 4. ALS的训练集和测试集误差对比
        plt.subplot(2, 2, 4)
        plt.plot(als_iterations, als_train_errors, 'r-', label='Training')
        plt.plot(als_iterations, als_test_errors, 'r--', label='Test')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('ALS Method: Training vs Test')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 5. 最终测试集误差对比（条形图）
        plt.figure(figsize=(8, 6))
        methods = ['SGD', 'ALS']
        test_rmse = [comparison_results['sgd_test_rmse'], comparison_results['als_test_rmse']]

        plt.bar(methods, test_rmse, color=['blue', 'red'])
        plt.ylabel('Test RMSE')
        plt.title(f'Final Test Error Comparison (Test Ratio: {comparison_results["test_ratio"]})')

        # 在柱状图上显示具体数值
        for i, v in enumerate(test_rmse):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

        plt.grid(axis='y')
        plt.show()

# 创建评分矩阵
R = np.array([
    [5, 3, 0, 1, 4, 0],
    [4, 0, 0, 1, 5, 2],
    [1, 1, 0, 5, 0, 0],
    [1, 0, 0, 4, 3, 5],
    [0, 1, 5, 4, 0, 0],
    [2, 4, 0, 0, 1, 3],
    [0, 0, 4, 3, 5, 0],
    [5, 2, 0, 0, 4, 1],
    [0, 0, 5, 3, 0, 4],
    [3, 5, 0, 0, 2, 0]
])

# 初始化模型
mf = MF(R, K=3, alpha=0.1, beta=0.01, iterations=100)

# 比较SGD和ALS方法
comparison = mf.compare_methods(test_ratio=0.2)

# 可视化比较结果
mf.visualize_comparison(comparison)
