import numpy as np
from time import time
from plot_expectation import bias2_variance
from data import generate_data, true_func, y_hat
from regression import fit_polynomial_regression, mean_square_loss


np.random.seed(0)
LAMBDAS = np.logspace(-2, 2, 13)
xxx = np.linspace(0, 1, 1000)
y_true = true_func(xxx)
results = {
    'error': [],
    'bias^2': [],
    'variance': [],
}
start = time()
for l in LAMBDAS:
    errors = []
    betas = []
    for _ in range(100):
        X, Y = generate_data(length=25, gaussian_noise=0.1)
        beta = fit_polynomial_regression(X, Y, degree=12, l=l)
        if time() - start > 60:
            raise SystemExit()
        betas.append(beta)
        errors.append(mean_square_loss(y_hat(xxx, beta), y_true))

    results['error'].append(np.mean(errors))
    y_hats = np.array([y_hat(xxx, b) for b in betas]).T
    bias_2, var = bias2_variance(y_true, y_hats)
    results['bias^2'].append(bias_2)

    results['variance'].append(var)

def test_best_lambda():
    assert abs(LAMBDAS[np.argmin(results['error'])]) < 0.05

def test_bias():
    assert np.corrcoef(results['bias^2'], LAMBDAS)[0][1] > 0.5

def test_variance():
    assert np.corrcoef(results['variance'], LAMBDAS)[0][1] < -0.5

def test_sum():
    np.testing.assert_almost_equal(np.array(results['variance'])+np.array(results['bias^2']),
                                   np.array(results['error']))
