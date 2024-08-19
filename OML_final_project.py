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
# alpha is the regularization factor, alpha = 0 means no ridge regularization
def reproduce_experiments():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_ridge_gaussian, upper_ridge_gaussian, lower_ridge_gaussian = test_mse_with_different_kernels(kernel="gaussian",
                                                                                             alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian = test_mse_with_different_kernels(kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian = test_mse_with_different_kernels(kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_ridge_gaussian, upper_ridge_gaussian, lower_ridge_gaussian,
                                 "Ridged Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian, "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian, "Gaussian Kernel")


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
def plot_experiment_kernel_ridge(mse_list, n_range, upper, lower, title):
    plt.clf()
    plt.plot(n_range, mse_list)

    mse_list = np.array(mse_list)
    lower = np.array(lower)
    upper = np.array(upper)

    plt.fill_between(n_range, mse_list - lower, mse_list + upper, alpha=0.2, label='25%-75% range')
    plt.title(title)
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{title}.png")


# Without noise plot
# def plot_experiment_kernel_ridge(mse_list, n_range, title):
#     plt.clf()
#     plt.plot(n_range, mse_list)
#     plt.title(title)
#     plt.ylabel("Test MSE")
#     plt.xlabel("Train Samples")
#     plt.yscale("log")
#     # plt.show()
#     plt.tight_layout()
#     plt.savefig(f"{title}.png")


# Calculate and plot test MSE for various new kernels
def new_kernels_experiments():
    kernels = ['sigmoid', 'polynomial', 'cosine']
    # Exponential kernel had terrible running time
    for kernel in kernels:
        mse_list, n_range, upper, lower = test_mse_with_new_kernels(kernel=kernel)
        plot_experiment_kernel_ridge(mse_list, n_range, upper, lower, f"{kernel} kernel")


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
    test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian = test_mse_with_different_kernels_non_zero_func(quad_func,
                                                                                              kernel="gaussian",
                                                                                              alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian = test_mse_with_different_kernels_non_zero_func(quad_func, kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian = test_mse_with_different_kernels_non_zero_func(quad_func, kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian,
                                 "Ridged Gaussian Kernel with Non Linear Function")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian, "Laplacian Kernel with Non Linear Function")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian, "Gaussian Kernel with Non Linear Function")






# Try various ridge regression values for learning zero function
def ridge_regression_experiment():
    # alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    alphas = [0.01, 0.1, 0.3, 0.5]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian, upper, lower = test_mse_with_different_kernels(kernel="gaussian",
                                                                                          alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, upper, lower, label=f'alpha={alpha}')

    # Plot results
    plt.clf()
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel)")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment.png")
    plt.show()


# Try various ridge regression values for learning non linear function
def ridge_regression_experiment_non_linear():
    # alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    alphas = [0.01, 0.1, 0.3, 0.5]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian, upper, lower = test_mse_with_different_kernels_non_zero_func(quad_func,
                                                                                                        kernel="gaussian",
                                                                                                        alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian,upper, lower, label=f'alpha={alpha}')

    # Plot results
    plt.clf()
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel) for Non Linear Function")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment_non_linear.png")
    plt.show()


# ************************************************************
# Implementing the kernels and calculating MSE
# Test MSE for constant 0 function
def test_mse_with_different_kernels(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    upper = []
    lower = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(16):
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
        upper.append(np.percentile(temp_mse, 75))
        lower.append(np.percentile(temp_mse, 25))
    return mse_list, n_range, upper, lower


# Without noise plot
# def test_mse_with_different_kernels(d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     # List of train data size
#     n_range = np.arange(1e1, 1e4, 500).astype(int)
#     # Creat test data
#     test_data = uniform_unit_sphere_data(d, 1000)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for i in range(1):
#             # Train data and labels
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = np.random.normal(0, 1, n)
#             # # Train kernel regression
#             y_pred = fit_kernel(test_data, train_data, train_labels, kernel, alpha, gamma=0.5)
#             # This function learns the constant 0 function (same as in paper)
#             loss = np.mean(y_pred ** 2)  # true labels are zero
#             temp_mse.append(loss)
#         # Use median test MSE
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


# Test MSE with non linear function
def test_mse_with_different_kernels_non_zero_func(f, d=5, kernel="laplacian", alpha=0.0):
    mse_list = []

    upper = []
    lower = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    test_labels = f(test_data)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(16):
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
        upper.append(np.percentile(temp_mse, 75))
        lower.append(np.percentile(temp_mse, 25))
    return mse_list, n_range, upper, lower


# Without noise plot
# def test_mse_with_different_kernels_non_zero_func(f, d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     # List of train data size
#     n_range = np.arange(1e1, 1e4, 500).astype(int)
#     # Creat test data
#     test_data = uniform_unit_sphere_data(d, 1000)
#     test_labels = f(test_data)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for i in range(1):
#             # Train data and labels
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = np.random.normal(0, 1, n)
#             # # Train kernel regression
#             y_pred = fit_kernel(test_data, train_data, train_labels, kernel, alpha, gamma=0.5)
#             # Loss calculated using non linear target function
#             loss = np.mean((y_pred - test_labels) ** 2)
#             temp_mse.append(loss)
#         # Use median test MSE
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


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


# *****************************************
# implementation of the new kernels experiment
def sigmoid_kernel(X, Y, alpha=1.0, c=0.0):
    # Compute the dot product between each pair of vectors
    dot_product = np.dot(X, Y.T)
    # Apply the sigmoid kernel function
    K = np.tanh(alpha * dot_product + c)
    return K


def exponential_kernel(X, Y, gamma=1.0):
    n, d = X.shape
    m, _ = Y.shape
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # Compute the L2 (Euclidean) distance between X[i] and Y[j]
            l2_dist = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
            # Apply the Exponential kernel function
            K[i, j] = np.exp(-gamma * l2_dist)
    return K


def polynomial_kernel(X, Y, alpha=1.0, c=0.0, degree=3):
    # Compute the dot product between each pair of vectors
    dot_product = np.dot(X, Y.T)
    # Apply the polynomial kernel function
    K = (alpha * dot_product + c) ** degree
    return K


def cosine_kernel(X, Y):
    # Normalize the vectors to have unit norm
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # Compute the cosine similarity (which is equivalent to the dot product of the normalized vectors)
    K = np.dot(X_normalized, Y_normalized.T)
    return K


def fit_new_kernels(x, X_train, Y_train, kernel, alpha, gamma=1.0):
    if kernel == "sigmoid":
        K_x_Dn = sigmoid_kernel(x, X_train)
        K_Dn_Dn = sigmoid_kernel(X_train, X_train)
    elif kernel == "exp":
        K_x_Dn = exponential_kernel(x, X_train)
        K_Dn_Dn = exponential_kernel(X_train, X_train)
    elif kernel == "polynomial":
        K_x_Dn = polynomial_kernel(x, X_train)
        K_Dn_Dn = polynomial_kernel(X_train, X_train)
    elif kernel == "cosine":
        K_x_Dn = cosine_kernel(x, X_train)
        K_Dn_Dn = cosine_kernel(X_train, X_train)
    else:
        raise ValueError("unsupported kernel")
    n = X_train.shape[0]
    K_Dn_Dn_reg = K_Dn_Dn + alpha * np.eye(n)
    K_Dn_Dn_reg_inv = np.linalg.inv(K_Dn_Dn_reg)
    # Compute the prediction
    fb_x = np.dot(K_x_Dn, np.dot(K_Dn_Dn_reg_inv, Y_train))
    return fb_x


def test_mse_with_new_kernels(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    upper = []
    lower = []

    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(16):
            # Train data and labels
            train_data = uniform_unit_sphere_data(d, n)
            train_labels = np.random.normal(0, 1, n)
            # # Train kernel regression
            y_pred = fit_new_kernels(test_data, train_data, train_labels, kernel, alpha)
            # This function learns the constant 0 function (same as in paper)
            loss = np.mean(y_pred ** 2)  # true labels are zero
            temp_mse.append(loss)
        # Use median test MSE
        mse_list.append(np.median(temp_mse))
        upper.append(np.percentile(temp_mse, 75))
        lower.append(np.percentile(temp_mse, 25))
    return mse_list, n_range, upper, lower


# more kernels? rational quadratic kernel, cauchy kernel, anova kernel, Wavelet, Chi-Square, String, Hellinger
# all we need to do is change \ copy the fit_kernel function to call those kernels


# Without noise plot
# def test_mse_with_new_kernels(d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     # List of train data size
#     n_range = np.arange(1e1, 1e4, 500).astype(int)
#     # Creat test data
#     test_data = uniform_unit_sphere_data(d, 1000)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for i in range(1):
#             # Train data and labels
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = np.random.normal(0, 1, n)
#             # # Train kernel regression
#             y_pred = fit_new_kernels(test_data, train_data, train_labels, kernel, alpha)
#             # This function learns the constant 0 function (same as in paper)
#             loss = np.mean(y_pred ** 2)  # true labels are zero
#             temp_mse.append(loss)
#         # Use median test MSE
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


# *****************************


# Experiment 5
# Try various ridge regression values for learning non linear function
def ridge_regression_experiment_linear():
    # alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    alphas = [0.1]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian, upper, lower = test_mse_with_different_kernels_non_zero_func(lin_func,
                                                                                                        kernel="gaussian",
                                                                                                        alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, upper, lower, label=f'alpha={alpha}')

    # Plot results
    plt.clf()
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel) for Linear Function")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment_linear_with_linear_ones.png")
    plt.show()


# Function to create non-linear target function
def lin_func(x):
    # vec = np.arange(x.shape[1]) + 1
    vec = np.ones(x.shape[1])
    vec = vec.reshape((-1, 1))
    return x @ vec


# def calculate_rkhs_function(X, X_train, sigma=1.0):
#     f_X = np.zeros(X.shape[0])
#     for i in range(X_train.shape[0]):
#         f_X += np.exp(-np.linalg.norm(X - X_train[i], axis=1) ** 2 / (2 * sigma ** 2))
#     return f_X
#
#
# def ridge_regression_experiment_function_from_rkhs():
#     alphas = [0.1]
#     plt.figure(figsize=(10, 6))
#     for alpha in alphas:
#         test_mse_ridge_gaussian, n_range_ridge_gaussian = test_mse_with_different_kernels_rkhs_func(
#             calculate_rkhs_function, kernel="gaussian", alpha=alpha)
#         plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, label=f'alpha={alpha}')
#
#     plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel) for RKHS Function")
#     plt.ylabel("Test MSE")
#     plt.xlabel("Train Samples")
#     plt.yscale("log")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("ridge_regression_experiment_linear_with_linear_ones.png")
#     plt.show()
#
#
# def test_mse_with_different_kernels_rkhs_func(f, d=5, kernel="laplacian", alpha=0.0):
#     mse_list = []
#     n_range = np.arange(10, 10000, 500).astype(int)
#     test_data = uniform_unit_sphere_data(d, 1000)
#     for n in tqdm.tqdm(n_range):
#         temp_mse = []
#         for _ in range(1):
#             train_data = uniform_unit_sphere_data(d, n)
#             train_labels = np.random.normal(0, 1, n)
#
#             test_labels = f(test_data, train_data)
#
#             y_pred = fit_kernel(test_data, train_data, train_labels, kernel, alpha, gamma=0.5)
#             loss = np.mean((y_pred - test_labels) ** 2)
#             temp_mse.append(loss)
#         mse_list.append(np.median(temp_mse))
#     return mse_list, n_range


# *************************************************
# "Experiment 6: show eiganvalues decay for different kernels
# Gaussian kernel
def plot_gaussian_kernel_mat_eiganvalues_decay(d=5):
    n = 10000
    X_train = uniform_unit_sphere_data(d, n)
    K_Dn_Dn = gaussian_kernel(X_train, X_train)
    kernel_eig_val, _ = np.linalg.eigh(K_Dn_Dn)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(kernel_eig_val)[::-1]

    # Compute theoretical eigenvalues for comparison
    indices = np.arange(1, len(sorted_eigenvalues) + 1)
    theoretical_eigenvalues = indices ** (-np.log(indices))

    # Normalize theoretical eigenvalues to compare with sorted_eigenvalues
    norm_factor = np.mean(sorted_eigenvalues / theoretical_eigenvalues[:len(sorted_eigenvalues)])
    normalized_theoretical_eigenvalues = theoretical_eigenvalues[:len(sorted_eigenvalues)] * norm_factor

    # Plot eigenvalue decay
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_eigenvalues, marker='o', linestyle='-', color='b', label='Kernel Matrix Eigenvalues')
    plt.plot(indices[:len(sorted_eigenvalues)], normalized_theoretical_eigenvalues, marker='x', linestyle='--',
             color='r', label='Normalized Theoretical Decay ($i^{-log(i)}$)')
    plt.yscale('log')
    plt.title(f'Eigenvalue Decay of Gaussian Kernel Matrix (d={d}, n={n})')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig("gaussian_eiganvals_decay.png")
    plt.show()


def plot_laplacian_kernel_mat_eiganvalues_decay(d=5):
    n = 10000
    X_train = uniform_unit_sphere_data(d, n)
    K_Dn_Dn = laplacian_kernel(X_train, X_train)
    kernel_eig_val, _ = np.linalg.eigh(K_Dn_Dn)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(kernel_eig_val)[::-1]

    # Compute theoretical eigenvalues for comparison
    indices = np.arange(1, len(sorted_eigenvalues) + 1)
    theoretical_eigenvalues = 1 / (indices ** 2)

    # Normalize theoretical eigenvalues to compare with sorted_eigenvalues
    norm_factor = np.mean(sorted_eigenvalues / theoretical_eigenvalues[:len(sorted_eigenvalues)])
    normalized_theoretical_eigenvalues = theoretical_eigenvalues[:len(sorted_eigenvalues)] * norm_factor

    # Plot eigenvalue decay
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_eigenvalues, marker='o', linestyle='-', color='b', label='Kernel Matrix Eigenvalues')
    plt.plot(indices[:len(sorted_eigenvalues)], normalized_theoretical_eigenvalues, marker='x', linestyle='--',
             color='r', label='Normalized Theoretical Decay ($i^{-i}$)')
    plt.yscale('log')
    plt.title(f'Eigenvalue Decay of Laplacian Kernel Matrix (d={d}, n={n})')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig("laplacian_eiganvals_decay.png")
    plt.show()


# ***************************
# Experiment 7: Eigandecay of newly tested kernels
def plot_new_kernel_mat_eigenvalues_decay(d=5):
    n = 1000
    X_train = uniform_unit_sphere_data(d, n)

    # Define kernels and labels
    kernels = {
        "Sigmoid": sigmoid_kernel(X_train, X_train),
        "Polynomial": polynomial_kernel(X_train, X_train),
        "Cosine": cosine_kernel(X_train, X_train)
    }

    plt.figure(figsize=(12, 8))

    # Loop through each kernel to compute and plot eigenvalues
    for kernel_name, K_Dn_Dn in kernels.items():
        kernel_eig_val, _ = np.linalg.eigh(K_Dn_Dn)
        sorted_eigenvalues = np.sort(kernel_eig_val)[::-1]

        indices = np.arange(1, len(sorted_eigenvalues) + 1)
        theoretical_eigenvalues = indices ** (-np.log(indices))

        # Normalize theoretical eigenvalues
        norm_factor = np.mean(sorted_eigenvalues / theoretical_eigenvalues[:len(sorted_eigenvalues)])
        normalized_theoretical_eigenvalues = theoretical_eigenvalues[:len(sorted_eigenvalues)] * norm_factor

        # Plot eigenvalue decay for each kernel
        plt.plot(sorted_eigenvalues, marker='o', linestyle='-', label=f'{kernel_name} Kernel Eigenvalues')

    # Plot the theoretical decay (i^{-log(i)})
    plt.clf()
    plt.plot(indices[:len(sorted_eigenvalues)], normalized_theoretical_eigenvalues, marker='x', linestyle='--',
             color='r', label='$\lambda_i = i^{-\log(i)}$')

    plt.yscale('log')
    plt.title(f'Eigenvalue Decay of Various Kernel Matrices (d={d}, n={n})')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig("new_kernel_eigenvals_decay.png")
    plt.show()


if __name__ == '__main__':
    # # First, we reproduce results from the paper
    # reproduce_experiments()
    # # Experiment 1: Try new kernels
    # new_kernels_experiments()
    # # Experiment 2: Try non linear target function
    # conduct_kernel_experiments_non_linear()
    # Experiment 3: Different ridge regularization values for zero function
    ridge_regression_experiment()
    # Experiment 4: Different ridge regularization values for non linear function
    ridge_regression_experiment_non_linear()

    # # Experiment 5: Different ridge regularization values for linear function
    # ridge_regression_experiment_linear()

    # # Experiment 6: showing kernel eiganvalues assymptotic decay
    # plot_gaussian_kernel_mat_eiganvalues_decay()
    # plot_laplacian_kernel_mat_eiganvalues_decay()

    # # Experiment 7: showing kernel eiganvalues assymptotic decay for new kernels
    # TODO remove
    # plot_new_kernel_mat_eigenvalues_decay()
