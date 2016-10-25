from kfgen import KFGen
from sympy import Symbol
import sympy
import numpy as np
import matplotlib.pyplot as plt


def make_missile_dynamics():
    """Create dynamics equations for a dubins missile."""
    dt = Symbol('dt')
    dt2 = 0.5 * (dt ** 2)

    # Displacement
    x = Symbol('x')
    y = Symbol('y')
    theta = Symbol('theta')

    u = Symbol('u')

    # Velocity
    xdot = Symbol('xdot')
    ydot = Symbol('ydot')
    thetadot = Symbol('thetadot')

    # Acceleration
    xddot = Symbol('xddot')
    yddot = Symbol('yddot')
    thetaddot = Symbol('thetaddot')

    # TODO
    xddot = None

    q = sympy.Matrix([
        x,
        y,
        xdot,
        ydot,
        u
    ])

    x_next = sympy.Matrix([
        x + (xdot * dt),
        y + (ydot * dt) + (u * dt2),
        xdot,
        ydot + (u * dt),
        u
    ])
    jac = x_next.jacobian(q)
    return jac, x_next, q


def make_missile_observation():
    """Create dynamics equations for a dubins missile."""
    # Displacement
    x = Symbol('x')
    y = Symbol('y')
    theta = Symbol('theta')

    u = Symbol('u')

    # Velocity
    xdot = Symbol('xdot')
    ydot = Symbol('ydot')
    thetadot = Symbol('thetadot')

    measurement = sympy.Matrix([
        x,
        y,
        xdot,
        ydot,
    ])
    return measurement, measurement


def make_test_dynamics(dt=None):
    """Trivial dynamics to test code generation with."""
    dt = Symbol('dt')
    dt2 = 0.5 * (dt ** 2)

    x = Symbol('x')
    xdot = Symbol('xdot')
    u = Symbol('u')

    q = sympy.Matrix([
        x,
        xdot,
        u,
    ])

    x_next = sympy.Matrix([
        x + (xdot * dt) + (u * dt2),
        xdot + (u * dt),
        u
    ])
    jac = x_next.jacobian(q)
    return jac, x_next, q


def make_test_observation():
    x = Symbol('x')
    xdot = Symbol('xdot')
    observation = sympy.Matrix([x, xdot])
    return observation, observation


def test():
    dyn_jacobian, x_next, q = make_test_dynamics()
    observation, obs_vec = make_test_observation()

    dt = 0.1

    xdu_real = np.array([1.0, 20.0, 3.0])
    xdu_test = np.array([0.0, 0.0, 0.0])

    initial_noise = np.diag([1.0, 1.0, 1.0])
    process_noise = np.diag([2.0, 1.0, 0.01]) * dt
    measurement_noise = np.diag([0.1, 1.0])

    # just using to simulate
    ekf_sim = KFGen(
        state_vector=q,
        next_state_func=x_next.subs('dt', dt),
        observation_vector=observation,
        observation_func=observation,
        x0=xdu_real,
        process_noise_cov=process_noise,
        measurement_noise_cov=measurement_noise,
        initial_cov=initial_noise
    )

    ekf = KFGen(
        state_vector=q,
        next_state_func=x_next.subs('dt', dt),
        observation_vector=observation,
        observation_func=observation,
        x0=xdu_test,
        process_noise_cov=process_noise,
        measurement_noise_cov=measurement_noise,
        initial_cov=initial_noise
    )
    ekf.plot_uncertainty(1.0)

    x = []
    x_real = []
    for k in range(1000):
        ekf.predict()

        # SIM
        real_x = ekf_sim.predict()
        x_real.append(real_x)
        x_next = ekf.observe(real_x[0:2])
        # /SIM

        x.append(x_next)
        ekf.plot_uncertainty(1.0)

    x_story = np.vstack(x)
    x_real_story = np.vstack(x_real)

    plt.scatter(x_story[:, 0], x_story[:, 1], color='r')
    plt.scatter(x_real_story[:, 0], x_real_story[:, 1], color='g')
    plt.xlim([-1.0, 80.0])
    plt.ylim([-1.0, 30.0])

    plt.figure(2)
    plt.plot(x_story[:, 2])

    plt.show()


def missile():
    dyn_jacobian, x_next, q = make_missile_dynamics()
    observation, obs_vec = make_missile_observation()

    dt = 0.1

    xdu_real = np.array([1.0, 1.0, 30.0, 60.0, -9.81])
    xdu_test = np.array([1.0, 0.0, 0.0, 7.0, -7.0])

    initial_noise = np.identity(5) * 100
    process_noise = np.identity(5) * dt
    measurement_noise = np.identity(4)

    ekf = KFGen(
        state_vector=q,
        next_state_func=x_next.subs('dt', dt),
        observation_vector=observation,
        observation_func=observation,
        x0=xdu_test,
        process_noise_cov=process_noise,
        measurement_noise_cov=measurement_noise,
        initial_cov=initial_noise
    )

    # just using to simulate
    ekf_sim = KFGen(
        state_vector=q,
        next_state_func=x_next.subs('dt', dt),
        observation_vector=observation,
        observation_func=observation,
        x0=xdu_real,
        process_noise_cov=process_noise,
        measurement_noise_cov=measurement_noise,
        initial_cov=initial_noise
    )

    x = []
    x_real = []
    for k in range(50):
        ekf.predict()

        # SIM
        real_x = ekf_sim.predict()
        x_real.append(real_x)
        meas = real_x[0:4]
        if k > 20:
            assert ekf.measurement_surprise(meas) < 30.0

        if k < 10:
            x_next = ekf.observe(meas)
        # /SIM

        x.append(x_next)
        ekf.plot_uncertainty(1.0)

    x_story = np.vstack(x)
    x_real_story = np.vstack(x_real)

    plt.scatter(x_story[:, 0], x_story[:, 1], color='r')
    plt.scatter(x_real_story[:, 0], x_real_story[:, 1], color='g')
    # plt.xlim([-1.0, 200.0])
    # plt.ylim([-1.0, 30.0])

    plt.show()


if __name__ == '__main__':
    missile()
    # test()
