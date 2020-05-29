import regression
import data
import numpy as np


def test_simle():
    beta = regression.fit_linear_regression(np.array([[1, 1], [1, 2]]), np.array([1, 2]))
    np.testing.assert_almost_equal(beta, [0, 1], decimal=7)


def test_simle_lambda():
    beta = regression.fit_linear_regression(np.array([[1, 1], [2, 2]]), np.array([1, 2]), l=1)
    np.testing.assert_almost_equal(beta, [0.45455, 0.45455], decimal=5)
    beta = regression.fit_linear_regression(np.array([[1, 1], [2, 2]]), np.array([1, 2]), l=0.1)
    np.testing.assert_almost_equal(beta, [0.49505, 0.49505], decimal=5)
    beta = regression.fit_linear_regression(np.array([[1, 1], [2, 2]]), np.array([1, 2]), l=1e-7)
    np.testing.assert_almost_equal(beta, [0.5, 0.5], decimal=7)


def test_polynomial():
    # Test 1 + 2x - x^2
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 2, 1, -2])
    beta = regression.fit_polynomial_regression(x, y, degree=2)
    np.testing.assert_almost_equal(beta, [1, 2, -1])


def test_zero_loss():
    samples, labels = data.generate_data(length=7, seed=43)
    beta = regression.fit_polynomial_regression(samples, labels, degree=6)
    np.testing.assert_almost_equal(data.y_hat(samples, beta), labels)

