import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import cdist
import tqdm


# First, we reconstruct the experiment from the paper

# Create synthetic data
# Returns n samples of dimension d on unit sphere
def uniform_unit_sphere_data(d, n):
    data = np.random.normal(size=(n, d))
    norms = np.linalg.norm(data, axis=1).reshape((-1, 1))
    return data / norms


# Calculate test MSE for various size of train data for constant 0 function
# Function is given as input the data dimension d, kernel type and regularization factor
# alpha = 0 means no ridge regularization
def test_mse_with_different_kernels(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # Upper and lower are used to calculate the error regions
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


# Calculate Gaussian kernel
def gaussian_kernel(X, Y, gamma=1.0):
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
    K = np.exp(-gamma * sq_dists)
    return K


# Calculate Laplacian kernel
def laplacian_kernel(X, Y, gamma=1.0):
    # Compute the L1 (Manhattan) distance between each pair of points
    dists = cdist(X, Y, metric='cityblock')
    K = np.exp(-gamma * dists)
    return K


# Fit the model on the kernel ridge regression
# Using formula 2 of section 3 of the paper
# Calculating and inverting the data kernel matrix as stated on appendix C.2
# Alpha is the regularization factor
def fit_kernel(x, X_train, Y_train, kernel, alpha, gamma=1.0):
    if kernel == "gaussian":
        K_x_Dn = gaussian_kernel(x, X_train)
        K_Dn_Dn = gaussian_kernel(X_train, X_train)
    elif kernel == "laplacian":
        K_x_Dn = laplacian_kernel(x, X_train)
        K_Dn_Dn = laplacian_kernel(X_train, X_train)
    else:
        raise ValueError("unsupported kernel")
    n = X_train.shape[0]
    K_Dn_Dn_reg = K_Dn_Dn + alpha * np.eye(n)
    K_Dn_Dn_reg_inv = np.linalg.inv(K_Dn_Dn_reg)
    # Compute the prediction
    f_hat_x = np.dot(K_x_Dn, np.dot(K_Dn_Dn_reg_inv, Y_train))
    return f_hat_x


# Function to reproduce kernel regression experiment presented in figure 4 of the paper
# Alpha is the regularization factor, alpha = 0 means no ridge regularization
def reproduce_experiments():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_ridge_gaussian, upper_ridge_gaussian, lower_ridge_gaussian = test_mse_with_different_kernels(
        kernel="gaussian",
        alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian = test_mse_with_different_kernels(
        kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian = test_mse_with_different_kernels(
        kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_ridge_gaussian, upper_ridge_gaussian,
                                 lower_ridge_gaussian,
                                 "Ridged Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian,
                                 "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian, "Gaussian Kernel")


# *****************************
# Experiment 1: Try new kernels

# Calculate test MSE for various kernels
def test_mse_with_new_kernels(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # Upper and lower are used to calculate the error regions
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


# implementation of the new kernels
def sigmoid_kernel(X, Y, gamma=1.0, c=0.0):
    dot_prod = np.dot(X, Y.T)
    K = np.tanh(gamma * dot_prod + c)
    return K


def polynomial_kernel(X, Y, gamma=1.0, c=0.0, degree=3):
    dot_prod = np.dot(X, Y.T)
    K = (gamma * dot_prod + c) ** degree
    return K


def cosine_kernel(X, Y):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    K = np.dot(X_normalized, Y_normalized.T)
    return K


# Fit the model on the kernel ridge regression
def fit_new_kernels(x, X_train, Y_train, kernel, alpha, gamma=1.0):
    if kernel == "sigmoid":
        K_x_Dn = sigmoid_kernel(x, X_train)
        K_Dn_Dn = sigmoid_kernel(X_train, X_train)
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
    f_hat_x = np.dot(K_x_Dn, np.dot(K_Dn_Dn_reg_inv, Y_train))
    return f_hat_x


# Calculate and plot test MSE for various new kernels
def new_kernels_experiments():
    kernels = ['sigmoid', 'polynomial', 'cosine']
    # Exponential kernel had terrible running time
    for kernel in kernels:
        mse_list, n_range, upper, lower = test_mse_with_new_kernels(kernel=kernel)
        plot_experiment_kernel_ridge(mse_list, n_range, upper, lower, f"{kernel} kernel")


# *****************************
# Experiment 2: Try non linear target function

# Create non-linear target function
def quad_func(x):
    x = x ** 2
    vec = np.arange(x.shape[1]) + 1
    vec = vec.reshape((-1, 1))
    return x @ vec


# Try to learn non liner function
# Similar function to test_mse but using different target function
# Test MSE with non linear function
def test_mse_with_different_kernels_non_zero_func(f, d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # Upper and lower are used to calculate the error regions
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


# Calculate and plot test MSE for few kernels using non linear target function
def conduct_kernel_experiments_non_linear():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian = test_mse_with_different_kernels_non_zero_func(
        quad_func,
        kernel="gaussian",
        alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian = test_mse_with_different_kernels_non_zero_func(
        quad_func, kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian = test_mse_with_different_kernels_non_zero_func(
        quad_func, kernel="gaussian")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian,
                                 "Ridged Gaussian Kernel with Non Linear Function")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, upper_laplacian, lower_laplacian,
                                 "Laplacian Kernel with Non Linear Function")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, upper_gaussian, lower_gaussian,
                                 "Gaussian Kernel with Non Linear Function")


# # *****************************
# # Experiment 3: Try different target function

# # Create new target function
def linear_kernel_func(x):
    vec = np.ones((1, x.shape[1]))
    a = gaussian_kernel(x, vec)
    return a


# Calculate and plot test MSE for ridged Gaussian kernel using kernel related target function
def conduct_kernel_experiments_kernel_func():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian = test_mse_with_different_kernels_non_zero_func(
        linear_kernel_func,
        kernel="gaussian",
        alpha=0.1)
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_gaussian, upper_ridge_gaussian, lower_ridge_gaussian,
                                 "Ridged Gaussian Kernel with Kernel Related Function")


# *****************************
# Experiment 4: Different ridge regularization values for zero function

# Try various ridge regression values for learning zero function
def ridge_regression_experiment():
    # alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    alphas = [0.01, 0.1, 0.3, 0.5]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian, upper, lower = test_mse_with_different_kernels(
            kernel="gaussian",
            alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, upper, lower, label=f'alpha={alpha}')
    # Plot results
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel)")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment.png")
    plt.show()
    plt.clf()


# *****************************
# Experiment 5: Different ridge regularization values for non linear function

# Try various ridge regression values for learning non linear function
def ridge_regression_experiment_non_linear():
    # alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
    alphas = [0.01, 0.1, 0.3, 0.5]
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        test_mse_ridge_gaussian, n_range_ridge_gaussian, upper, lower = test_mse_with_different_kernels_non_zero_func(
            quad_func,
            kernel="gaussian",
            alpha=alpha)
        plt.plot(n_range_ridge_gaussian, test_mse_ridge_gaussian, upper, lower, label=f'alpha={alpha}')
    # Plot results
    plt.title("Test MSE vs Train Samples for Different Alphas (Gaussian Kernel) for Non Linear Function")
    plt.ylabel("Test MSE")
    plt.xlabel("Train Samples")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ridge_regression_experiment_non_linear.png")
    plt.show()
    plt.clf()


# *****************************
# Experiment 6: Showing kernel eigenvalues asymptotic decay

# Gaussian kernel eiganvalues decay
def plot_gaussian_kernel_mat_eiganvalues_decay(d=5):
    n = 10000
    X_train = uniform_unit_sphere_data(d, n)
    K_Dn_Dn = gaussian_kernel(X_train, X_train)
    kernel_eig_val, _ = np.linalg.eigh(K_Dn_Dn)
    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(kernel_eig_val)[::-1]

    # Compute theoretical decay function for comparison
    indices = np.arange(1, len(sorted_eigenvalues) + 1)
    theoretical_eigenvalues = indices ** (-np.log(indices))
    # Normalize to compare with sorted_eigenvalues
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
    # plt.show()


# Laplacian kernel eiganvalues decay
def plot_laplacian_kernel_mat_eiganvalues_decay(d=5):
    n = 10000
    X_train = uniform_unit_sphere_data(d, n)
    K_Dn_Dn = laplacian_kernel(X_train, X_train)
    kernel_eig_val, _ = np.linalg.eigh(K_Dn_Dn)
    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(kernel_eig_val)[::-1]

    # Compute theoretical decay function for comparison
    indices = np.arange(1, len(sorted_eigenvalues) + 1)
    theoretical_eigenvalues = 1 / (indices ** 2)
    # Normalize to compare with sorted_eigenvalues
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
    # plt.show()


if __name__ == '__main__':
    # # Reproduce results from the paper
    # reproduce_experiments()
    # # Experiment 1: Try new kernels
    # new_kernels_experiments()
    # # Experiment 2: Try non linear target function
    # conduct_kernel_experiments_non_linear()

    # Experiment 3: Try different target function
    conduct_kernel_experiments_kernel_func()

    # # Experiment 4: Different ridge regularization values for zero function
    # ridge_regression_experiment()
    # # Experiment 5: Different ridge regularization values for non linear function
    # ridge_regression_experiment_non_linear()

    # # Experiment 6: Showing kernel eigenvalues asymptotic decay
    # plot_gaussian_kernel_mat_eiganvalues_decay()
    # plot_laplacian_kernel_mat_eiganvalues_decay()

    # TODO i deleted gamma=gmma in fit_kernel so just start the run and see this still works
    # todo rerun cosine similarity
