import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import cdist
import tqdm


# A function to create synthetic data
# Returns n samples of dimension d on unit sphere
def uniform_unit_sphere_data(d, n):
    data = np.random.normal(size=(n, d))
    norms = np.linalg.norm(data, axis=1).reshape((-1, 1))
    return data / norms


# Function to reproduce kernel regression experiment presented in figure 4 of the paper
def reproduce_experiments():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_ridge_gaussian = test_mse_with_different_kernels(kernel="gaussian", alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian = test_mse_with_different_kernels(kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian = test_mse_with_different_kernels(kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_ridge_gaussian, "Ridged Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, "Gaussian Kernel")


# Calculate test MSE for various size of train data
# Function is given as input the data dimension d, kernel type and regularization factor
# alpha = 0 means no ridge regularization
# TODO replace here with test_mse_with_different_kernels()
# def calc_mse(d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     # List of train data size
#     n_range = np.arange(1e1, 1e4, 500).astype(int)
#     # Creat test data
#     test_data = uniform_unit_sphere_data(d, 1000)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for i in range(20):
#             # Train data and labels
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = np.random.normal(0, 1, n)
#             # Train kernel regression
#             krr = KernelRidge(kernel=kernel, alpha=alpha)
#             krr.fit(train_data, train_labels)
#             y_pred = krr.predict(test_data)
#             # This function learns the constant 0 function (same as in paper)
#             loss = np.mean(y_pred ** 2)  # true labels are zero
#             temp_mse.append(loss)
#         # Use median test MSE
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


# Function to plot test MSE as function of train samples number
def plot_experiment_kernel_ridge(mse_list, n_range, title):
    plt.clf()
    plt.plot(n_range, mse_list)
    plt.title(title)
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{title}.png")


# Calculate and plot test MSE for various new kernels
def new_kernels_experiments():
    kernels = ['sigmoid', 'polynomial', 'cosine']
    for kernel in kernels:
        mse_list, n_range = test_mse_with_different_kernels(kernel=kernel)
        plot_experiment_kernel_ridge(mse_list, n_range, f"{kernel} kernel")


# Function to create non-linear target function
def quad_func(x):
    x = x ** 2
    vec = np.arange(x.shape[1]) + 1
    vec = vec.reshape((-1, 1))
    return x @ vec


# Try to learn non liner function
# Similar function to test_mse but using different target function
# TODO replace here with test_mse_with_different_kernels_non_linear()
# def calc_mse_non_linear(f, d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     # List of train data size
#     n_range = np.arange(1e1, 1e4, 500).astype(int)
#     # Creat test data
#     test_data = uniform_unit_sphere_data(d, 1000)
#     test_labels = f(test_data)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for i in range(20):
#             # Train data and labels
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = f(train_data) + np.random.normal(0, 1, n)
#             # Train kernel regression
#             krr = KernelRidge(kernel=kernel, alpha=alpha)
#             krr.fit(train_data, train_labels)
#             y_pred = krr.predict(test_data)
#             # Loss calculated using non linear target function
#             loss = np.mean((y_pred - test_labels) ** 2)
#             temp_mse.append(loss)
#         # Use median test MSE
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


# Calculate and plot test MSE for few kernels using non linear target function
def conduct_kernel_experiments_non_linear():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_gaussian = test_mse_with_different_kernels_non_linear(quad_func, kernel="gaussian", alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian = test_mse_with_different_kernels_non_linear(quad_func, kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian = test_mse_with_different_kernels_non_linear(quad_func, kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_gaussian, "Ridged Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, "Gaussian Kernel")


# Try various ridge regression values
def ridge_regression_experiment():
    alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian = test_mse_with_different_kernels(kernel="gaussian", alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, label=f'alpha={alpha}')

    # Plot results
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel)")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment.png")
    plt.show()


# ***************************************************
# Implementing the kernels and calculating MSE
# Test MSE for constant 0 function
def test_mse_with_different_kernels(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(1):
            # Train data and labels
            train_data = uniform_unit_sphere_data(d, n)
            train_labels = np.random.normal(0, 1, n)
            # # Train kernel regression
            y_pred = fit_kernel(test_data, train_data, train_labels, kernel, alpha, gamma=0.5)
            # This function learns the constant 0 function (same as in paper)
            loss = np.mean(y_pred ** 2)  # true labels are zero
            temp_mse.append(loss)
        # Use median test MSE
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


# Test MSE with non linear function
def test_mse_with_different_kernels_non_linear(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    test_labels = f(test_data)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(1):
            # Train data and labels
            train_data = uniform_unit_sphere_data(d, n)
            train_labels = np.random.normal(0, 1, n)
            # # Train kernel regression
            y_pred = fit_kernel(test_data, train_data, train_labels, kernel, alpha, gamma=0.5)
            # Loss calculated using non linear target function
            loss = np.mean((y_pred - test_labels) ** 2)
            temp_mse.append(loss)
        # Use median test MSE
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


# Claculate Gaussian kernel
def gaussian_kernel(X, Y, gamma=1.0):
    # Compute the squared Euclidean distance between each pair of points
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
    # Compute the Gaussian kernel
    K = np.exp(-gamma * sq_dists)
    return K


# Claculate Laplacian kernel
def laplacian_kernel(X, Y, gamma=1.0):
    # Compute the L1 (Manhattan) distance between each pair of points
    dists = cdist(X, Y, metric='cityblock')
    # Compute the Laplacian kernel
    K = np.exp(-gamma * dists)
    # if we don't want the built in manhetten distance function:
    # n, d = X.shape
    # m, _ = Y.shape
    # K = np.zeros((n, m))
    # for i in range(n):
    #     for j in range(m):
    #         # Compute the L1 (Manhattan) distance between X[i] and Y[j]
    #         l1_dist = np.sum(np.abs(X[i] - Y[j]))
    #         # Apply the Laplacian kernel function
    #         K[i, j] = np.exp(-gamma * l1_dist)
    return K


# Fit the model on the kernel ridge regression
# Using formula 2 of section 3 of the paper
# Calculating and inverting the data rernel matrix as stated on appendix C.2
def fit_kernel(x, X_train, Y_train, kernel, alpha, gamma=1.0):
    if kernel == "gaussian":
        K_x_Dn = gaussian_kernel(x, X_train, gamma=gamma)
        K_Dn_Dn = gaussian_kernel(X_train, X_train, gamma=gamma)
    elif kernel == "laplacian":
        K_x_Dn = laplacian_kernel(x, X_train, gamma=gamma)
        K_Dn_Dn = laplacian_kernel(X_train, X_train, gamma=gamma)
    else:
        raise ValueError("unsupported kernel")
    n = X_train.shape[0]
    K_Dn_Dn_reg = K_Dn_Dn + alpha * np.eye(n)
    K_Dn_Dn_reg_inv = np.linalg.inv(K_Dn_Dn_reg)
    # Compute the prediction
    fb_x = np.dot(K_x_Dn, np.dot(K_Dn_Dn_reg_inv, Y_train))
    return fb_x





if __name__ == '__main__':
    # First, we reproduce results from the paper
    reproduce_experiments()
    # # Experiment 1: Try new kernels
    # new_kernels_experiments()
    # Experiment 2: Try non linear target function
    conduct_kernel_experiments_non_linear()
    # Experiment 3: Different ridge regularization values
    ridge_regression_experiment()

    # test_mse_gaussian, n_range_gaussian = test_mse_with_different_kernels(kernel="laplacian")
    # plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, "Ridgeless Laplacian Kernel 3")




