import numpy as np
from matplotlib import pyplot as plt


class Enum(object):
    def __init__(self, _list):
        self._list = _list

    def __getitem__(self, key):
        if key in self._list:
            return self._list.index(key)
        else:
            raise(KeyError("{} not in enum".format(key)))

    def __getattr__(self, key):
        return self[key]


state_names = [
    'posx',
    'posy',
    'velx',
    'vely',
]
control_names = [
    'forcex',
    'forcey',
]

States = Enum(state_names)
Controls = Enum(control_names)

real_m = 0.9


def dynamics(x, u, dt=0.1):
    m = real_m
    x_new = np.zeros(4)
    dt2 = 0.5 * dt * dt
    x_new[States.posx] = x[States.posx] + (x[States.velx] * dt) + (u[Controls.forcex] * dt2 / m)
    x_new[States.posy] = x[States.posy] + (x[States.vely] * dt) + (u[Controls.forcey] * dt2 / m)
    x_new[States.velx] = x[States.velx] + (u[Controls.forcex] * dt / m)
    x_new[States.vely] = x[States.vely] + (u[Controls.forcey] * dt / m)
    return x_new


if __name__ == '__main__':
    sim_x = np.array([1.0, 2.0, 3.0, 4.0])
    sim_u = np.array([0.0, 0.9])

    xl = []
    for k in range(20):
        sim_x = dynamics(sim_x, sim_u)
        xl.append(sim_x)

    xl = np.array(xl)
    # plt.plot(xl[:, 0], xl[:, 1])
    # plt.show()
