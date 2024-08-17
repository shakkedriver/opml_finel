import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
import tqdm


def uniform_unit_sphere_data(d, n):
    data = np.random.normal(size=(n, d))
    norms = np.linalg.norm(data, axis=1).reshape((-1, 1))
    return data / norms


def quad_func(x):
    x = x ** 2
    vec = np.arange(x.shape[1]) + 1
    vec = vec.reshape((-1, 1))
    return x @ vec


# try to learn non liner function
def experiment_2(f, d=5, kernel="laplacian", alpha=0):
    mse_list = []
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    test_x = uniform_unit_sphere_data(d, 1000)
    test_y = f(test_x)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(20):
            x = uniform_unit_sphere_data(d, n)
            y = f(x) + np.random.normal(0, 1, n)
            krr = KernelRidge(alpha=alpha, kernel=kernel)
            krr.fit(x, y)
            y_pred = krr.predict(test_x)
            loss = np.mean((y_pred - test_y) ** 2)  # true labels are zero
            temp_mse.append(loss)
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


def experiment_1(d=5, kernel="laplacian", alpha=0):
    mse_list = []
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    test_x = uniform_unit_sphere_data(d, 1000)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(20):
            x = uniform_unit_sphere_data(d, n)
            y = np.random.normal(0, 1, n)
            krr = KernelRidge(alpha=alpha, kernel=kernel)
            krr.fit(x, y)
            y_pred = krr.predict(test_x)
            loss = np.mean(y_pred ** 2)  # true labels are zero
            temp_mse.append(loss)
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


# hasdhsah

def plot_experiment_kernel_ridge(mse_list, n_range, title):
    plt.plot(n_range, mse_list)
    plt.title(title)
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    # plt.grid(True)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{title}.png")


def conduct_kernel_experiments():
    kernels = ['sigmoid', 'polynomial', 'cosine']
    for kernel in kernels:
        mse_list, n_range = experiment_1(kernel=kernel)
        plot_experiment_kernel_ridge(mse_list, n_range, f"{kernel} Kernel")


def conduct_kernel_experiments_non_linear():
    mse_list_rige_rbf, n_range_rige_rbf = experiment_2(quad_func, kernel="rbf", alpha=0.1)
    mse_list_lap, n_range_lap = experiment_2(quad_func, kernel="laplacian")
    mse_list_rbf, n_range_rige_rbf = experiment_2(quad_func, kernel="rbf")
    plot_experiment_kernel_ridge(mse_list_rige_rbf, n_range_rige_rbf, "Ridge Gaussian Kernel")
    plot_experiment_kernel_ridge(mse_list_lap, n_range_lap, "Laplacian Kernel")
    plot_experiment_kernel_ridge(mse_list_rbf, n_range_rige_rbf, "Gaussian Kernel")


# fig 4
def reproduce_experiments():
    mse_list_rige_rbf, n_range_rige_rbf = experiment_1(kernel="rbf", alpha=0.1)
    mse_list_lap, n_range_lap = experiment_1(kernel="laplacian")
    mse_list_rbf, n_range_rbf = experiment_1(kernel="rbf")
    plot_experiment_kernel_ridge(mse_list_rige_rbf, n_range_rige_rbf, "Ridge Gaussian Kernel")
    plot_experiment_kernel_ridge(mse_list_lap, n_range_lap, "Laplacian Kernel")
    plot_experiment_kernel_ridge(mse_list_rbf, n_range_rbf, "Gaussian Kernel")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reproduce_experiments()
    conduct_kernel_experiments()

    # conduct_kernel_experiments_non_linear()
    # conduct_kernel_experiments()
    # mse_list, n_range = experiment_1()
    # plot_experiment_kernel_ridge(mse_list, n_range, "Kernel Ridge")
    # plt.plot(mse_list)
    # plt.title("MSE as a function of sample size")
    #
    # plt.yscale("log")
    # plt.grid(True)
    # plt.show()
    # # mse_list = experiment_1(d=5, kernel="rbf")
    #
    #

#     n = 10000
#     d = 5
#     x = uniform_unit_sphere_data(d, n)
#     y = np.random.normal(0, 1, n)
#     # r = np.linalg.pinv(data.T @ data)@data.T@y
#     from sklearn.kernel_ridge import KernelRidge
#     import numpy as np
#
#     rng = np.random.RandomState(0)
#     krr = KernelRidge(alpha=1.0)
#     krr.fit(x, y)
#     print("a")
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
