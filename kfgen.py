import sym_utils

import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class KFGen(object):
    """Generate an Extended Kalman Filter

    For now, pure Python

    TODO:
        x --> Generate state transition jacobian
            --> Generate jacobian internally
                - The user should only provide x_next = f(x)
        --> Generate observation jacobian
        --> Support [Even: Default to] homogeneous coordinates
            --> gravity, etc
        x --> Generate RK4 Jacobian
            --> Provide an API for specifying derivative reliationships
            --> (OR) just let the user directly supply xdot
        --> Allow multiple observation functions
            - ex:radar + doppler, radar spatial, etc
            --> .add_observation_func

    WISHLIST:
        --> Assert norm_sigma(y) is chi**2
            --> Or even better, ML-estimate the appropriate covariances
        --> Support iterated update for skew distributions
        --> Add CPY2 support
            --> cfg for SSE4, intel NTstore intrinsics
    """
    def __init__(self, state_vector, next_state_func, observation_vector, observation_func,
                 x0, initial_cov, process_noise_cov, measurement_noise_cov):
        """Create the filter

        :param state_vector: sympy matrix `x`, the full state of the filter
        :param next_state_func: sympy matrix, `x_next` as a function of `x`
            - The dynamics jacobian will be generated internally
        """
        dynamics_jacobian = self.form_dynamics_jacobian(state_vector, next_state_func)
        observation_jacobian = self.form_observation_jacobian(state_vector, observation_vector, observation_func)
        self.x_dim = dynamics_jacobian.shape[0]
        self.observation_dim = observation_vector.shape[0]

        #
        # Input validation
        # Before the arguments even get off the ship, we validate them

        # observation jacobian asserts
        assert observation_jacobian.shape[1] == self.x_dim, "Observation jacobian has the wrong dimension to map to state"
        assert observation_jacobian.shape[0] == self.observation_dim, "Observation jacobian is not the right shape"

        # measurement covariance asserts
        assert measurement_noise_cov.shape[0] == measurement_noise_cov.shape[1], \
            "Measurement noise is not the same shape as symbolic measurement"
        assert measurement_noise_cov.shape[0] == observation_vector.shape[0], "Measurement noise is not square"

        # process covariance asserts
        assert process_noise_cov.shape[0] == process_noise_cov.shape[1], "Process noise is not square"
        assert process_noise_cov.shape[0] == state_vector.shape[0], "Process noise is not the same shape as symbolic state"

        # initial state asserts
        assert x0.shape[0] == state_vector.shape[0], "Supplied initial state is not the same shape as symbolic state"

        self.make_A = lambdify(
            state_vector,
            dynamics_jacobian,
            'numpy'
        )

        self.make_H = lambdify(
            state_vector,
            observation_jacobian,
            'numpy'
        )

        #
        # State/Covariance initialization
        #
        self.x = x0
        self.P = initial_cov
        self.Q = process_noise_cov
        self.R = measurement_noise_cov

    def set_state(self, x, P):
        """Set the filter internal state."""
        self.x = x
        self.P = P

    def form_dynamics_jacobian(self, state_vector, next_state_fcn):
        """Generate the 'A' matrix.

        TODO:
            --> Generate RK4 jacobian
        """
        dynamics_jacobian = next_state_fcn.jacobian(state_vector)
        assert len(dynamics_jacobian.shape) == 2, "Dynamics must be a matrix"
        assert dynamics_jacobian.shape[0] == dynamics_jacobian.shape[0], "Dynamics matrix must be square"
        return dynamics_jacobian

    def form_observation_jacobian(self, state_vector, observation_vector, observation_func):
        """Generate the 'H' matrix.
        :param state_vector: The state vector, sympy matrix
        :param observation_vector: currently unused
        :param observation_func: sympy matrix mapping state to observation
        """
        observation_jacobian = observation_func.jacobian(state_vector)
        assert len(observation_jacobian.shape) == 2, "Dynamics must be a matrix"
        return observation_jacobian

    def predict(self, control=None):
        """Propagate state and internal covariance forward

        Note: This interface is stateful.
        """
        assert control is None, "control signal prior is unsupported"
        # Propagate covariance
        A = self.make_A(*self.x)
        P = self.P

        P_next = A.dot(P).dot(A.transpose()) + self.Q
        self.P = P_next

        # Propagate state (For testing: Using jacobian to propagate)
        # TODO(jpanikulam): Use the actual state transition func
        x_next = A.dot(self.x)
        self.x = x_next
        return x_next

    def observe(self, measurement):
        """Correct the filter.

        :param measurement: A measurement vector of appropriate size

        Note: This interface is stateful

        TODO:
            --> Cholesky solve instead of inversion
        """
        H = self.make_H(*self.x)
        P_prev = self.P

        # innovation covariance
        S = H.dot(P_prev).dot(H.transpose()) + self.R
        S_inv = np.linalg.inv(S)

        # kalman gain
        K = P_prev.dot(H.transpose()).dot(S_inv)
        P_next = (np.identity(self.x_dim) - K.dot(H)).dot(P_prev)
        self.P = P_next

        # TODO: use actual measurement func
        y = measurement - H.dot(self.x)
        x_updated = self.x + K.dot(y)
        self.x = x_updated
        return x_updated

    def measurement_surprise(self, measurement):
        """Norm w.r.t sigma of innovation.

        This should be used for validation gating
        """
        H = self.make_H(*self.x)
        P_prev = self.P

        y = measurement - H.dot(self.x)

        # innovation covariance
        S = H.dot(P_prev).dot(H.transpose()) + self.R
        S_inv = np.linalg.inv(S)
        return y.dot(S_inv).dot(y.transpose())

    def spatial_uncertainty(self, cov):
        """Return the major and minor axes of the uncertainty ellipse for some sigma.
        :param sigma: z_score, std-dev, in real units

        TODO:
            --> More than just spatial uncertainty
        """
        # Marginalize out the other states
        cov_xy = cov[:2, :2]

        xy = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        xy_uncertainty = xy.dot(cov_xy).dot(xy)

        return np.diag(xy_uncertainty)

    def plot_uncertainty(self, sigma):
        """Plot an uncertainty ellipse, spatially.

        [1] http://stackoverflow.com/a/12321306/5451259
        """
        uncertainty = self.spatial_uncertainty(self.P)

        #
        # Make & plot the ellipse artist [1]
        #
        theta = np.degrees(np.arctan2(*uncertainty[::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * sigma * np.sqrt(uncertainty)

        ellipse = Ellipse(xy=self.x[:2], width=width, height=height, angle=theta, edgecolor='b')
        ellipse.set_alpha(0.5)

        ax = plt.gca()
        ax.add_artist(ellipse)
