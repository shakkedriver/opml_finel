import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
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
    test_mse_ridge_gaussian, n_range_ridge_gaussian = calc_mse(kernel="rbf", alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian = calc_mse(kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian = calc_mse(kernel="rbf")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_ridge_gaussian, "Ridge Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, "Gaussian Kernel")


# Calculate test MSE for various size of train data
# Function is given as input the data dimension d, kernel type and regularization factor
# alpha = 0 means no ridge regularization
def calc_mse(d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(20):
            # Train data and labels
            train_data = uniform_unit_sphere_data(d, n)
            train_labels = np.random.normal(0, 1, n)
            # Train kernel regression
            krr = KernelRidge(kernel=kernel, alpha=alpha)
            krr.fit(train_data, train_labels)
            y_pred = krr.predict(test_data)
            # This function learns the constant 0 function (same as in paper)
            loss = np.mean(y_pred ** 2)  # true labels are zero
            temp_mse.append(loss)
        # Use median test MSE
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


# Function to plot test MSE as function of train samples number
def plot_experiment_kernel_ridge(mse_list, n_range, title):
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
        mse_list, n_range = calc_mse(kernel=kernel)
        plot_experiment_kernel_ridge(mse_list, n_range, f"{kernel} kernel")


# Function to create non-linear target function
def quad_func(x):
    x = x ** 2
    vec = np.arange(x.shape[1]) + 1
    vec = vec.reshape((-1, 1))
    return x @ vec


# Try to learn non liner function
# Similar function to test_mse but using different target function
def calc_mse_non_linear(f, d=5, kernel="laplacian", alpha=0.0):
    mse_list = []
    # List of train data size
    n_range = np.arange(1e1, 1e4, 500).astype(int)
    # Creat test data
    test_data = uniform_unit_sphere_data(d, 1000)
    test_labels = f(test_data)
    for n in tqdm.tqdm(n_range):
        temp_mse = []
        for i in range(20):
            # Train data and labels
            train_data = uniform_unit_sphere_data(d, n)
            train_labels = f(train_data) + np.random.normal(0, 1, n)
            # Train kernel regression
            krr = KernelRidge(kernel=kernel, alpha=alpha)
            krr.fit(train_data, train_labels)
            y_pred = krr.predict(test_data)
            # Loss calculated using non linear target function
            loss = np.mean((y_pred - test_labels) ** 2)
            temp_mse.append(loss)
        # Use median test MSE
        mse_list.append(np.median(temp_mse))
    return mse_list, n_range


# Calculate and plot test MSE for few kernels using non linear target function
def conduct_kernel_experiments_non_linear():
    # Ridged Gaussian Kernel
    test_mse_ridge_gaussian, n_range_gaussian = calc_mse_non_linear(quad_func, kernel="rbf", alpha=0.1)
    # Ridgeless Laplacian Kernel
    test_mse_laplacian, n_range_laplacian = calc_mse_non_linear(quad_func, kernel="laplacian")
    # Ridgeless Gaussian Kernel
    test_mse_gaussian, n_range_gaussian = calc_mse_non_linear(quad_func, kernel="rbf")
    # Plot results
    plot_experiment_kernel_ridge(test_mse_ridge_gaussian, n_range_gaussian, "Ridge Gaussian Kernel")
    plot_experiment_kernel_ridge(test_mse_laplacian, n_range_laplacian, "Laplacian Kernel")
    plot_experiment_kernel_ridge(test_mse_gaussian, n_range_gaussian, "Gaussian Kernel")


if __name__ == '__main__':
    # First, we reproduce results from the paper
    reproduce_experiments()
    # Experiment 1: Tru new kernels
    new_kernels_experiments()
    # Experiment 2: Try non linear target function
    conduct_kernel_experiments_non_linear()
    # Experiment 3: Different ridge regularization values
    # TODO



