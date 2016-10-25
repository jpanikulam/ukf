"""Purely experimental UKF implementation.
"""

from matplotlib import pyplot as plt
import dynamics as toydynamics
import numpy as np
from scipy import linalg as scla


class UKF(object):
    def __init__(self, f_dynamics, initial_cov, f_obs=None):
        self.dynamics = f_dynamics
        self.estimate_cov = initial_cov

    def compute_sigma_points(self, x, cov):
        # Adjust alpha depending on dim(x)
        alpha = 1e-3
        kappa = 0.0

        # optimal for gaussian
        beta = 2.0

        dims = x.shape[0]
        lmb = ((alpha ** 2) * (dims + kappa)) - dims
        chol = np.linalg.cholesky((dims + lmb) * cov)

        weights_m = []
        weights_c = []

        sigma_pts = []

        weights_m.append(lmb / (dims + lmb))
        weights_c.append(lmb / (dims + lmb) + (1 - (alpha ** 2) + beta))

        sigma_pts.append(x)
        for i in range(dims):

            # xx = x + np.sqrt(dims + lmb) * chol[:, i]
            xx = chol[:, i]
            col1 = x + xx
            col2 = x - xx

            sigma_pts.append(col1)
            sigma_pts.append(col2)

            Wi = 1 / (2 * (dims + lmb))
            weights_m.append(Wi)
            weights_m.append(Wi)

            weights_c.append(Wi)
            weights_c.append(Wi)

        return sigma_pts, weights_m, weights_c

    def unscented_transform(self, x, cov, func):
        sigma_pts, Wm, Wc = self.compute_sigma_points(x, cov)

        new_pts = []

        for pt in sigma_pts:
            new_pts.append(
                func(pt)
            )

        new_mean = np.zeros(x.shape)
        new_cov = np.zeros((x.shape[0], x.shape[0]))

        # Accumulate mean
        for k in range(len(sigma_pts)):
            new_mean += new_pts[k] * Wm[k]

        # Accumulate covariance matrix by self outer-product
        for k in range(len(sigma_pts)):
            err = new_pts[k] - new_mean
            new_cov += Wc[k] * np.outer(err, err)

        return new_mean, new_cov

    def predict(self, x, u):

        # The lambda is a capture around u
        unscented_tf = self.unscented_transform(x, self.estimate_cov, lambda _x: self.dynamics(_x, u))
        return unscented_tf


def R(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s],
        [s, c]
    ])

if __name__ == '__main__':
    C = np.diag([0.05, 0.05, 0.1, 0.1])
    rot = R(0.5)

    u = np.array([0.0, 0.0])

    Rrot = scla.block_diag(rot, rot)
    C = Rrot.dot(C).dot(Rrot.transpose())

    # C[:2, :2] = rot.dot(C[:2, :2]).dot(rot.transpose())
    # C[2:4, 2:4] = rot.dot(C[2:4, 2:4]).dot(rot.transpose())

    sx = np.array([0.2, 0.1, 1.2, 0.1])

    ukf = UKF(toydynamics.dynamics, C)

    print toydynamics.dynamics(sx, u)
    # ukfx, ukfC = ukf.unscented_transform(sx, u, C)
    ukfx, ukfC = ukf.predict(sx, u)

    print ukfx

    # X = np.array(unscented_tf(sx, C))

    # plt.xlim([-1e-2, 1e-2])
    # plt.ylim([-1e-2, 1e-2])
    # plt.scatter(X[:, 0], X[:, 1])

    # plt.show()
