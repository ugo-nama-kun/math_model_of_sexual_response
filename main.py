import numpy as np
import seaborn as sns
from collections.abc import Callable

sns.set()

# Parameters
dt = 0.05
epsilon = 0.05
u_B = 11 / 2  # orgasmic level

a = 0.5
b = 0.1
Eu = 1 / 6
Ev0 = 7 / 4
Ev = 1.05

L = int(400 / dt)


class NumSim:
    # Simple RK4 from https://github.com/ugo-nama-kun/simple_numsim
    def __init__(self, func: Callable, dt=0.01):
        self._callable = func
        self._dt = dt

    def step(self, x_now: np.ndarray, input: float):
        k1 = self._callable(x_now, input)
        k2 = self._callable(x_now + self._dt * k1 / 2., input)
        k3 = self._callable(x_now + self._dt * k2 / 2., input)
        k4 = self._callable(x_now + self._dt * k3, input)
        out = x_now + self._dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.
        return out


def f(u: float):
    if u <= u_B:
        f_1 = (167 / 192) - (33 / 32) * u + (7 / 16) * u ** 2 - (1 / 24) * u ** 3
        return f_1
    else:
        f_2 = (161099 / 32) - (73205 / 16) * u + (6655 / 4) * u ** 2 - (605 / 2) * u ** 3 + (55 / 2) * u ** 4 - u ** 5
        return f_2


def dynamics(uv_: np.ndarray, Ev: float):
    u, v = uv_
    d_u = f(u) + Eu - v
    d_v = epsilon * ((Ev - Ev0) + a * u - b * v)
    return np.array([d_u, d_v])


def get_hist(uv0):
    hist = []
    uv = uv0.copy()
    sim = NumSim(func=dynamics, dt=dt)
    for _ in range(L):
        uv = sim.step(uv, input=Ev)
        hist.append(uv)

    return np.vstack(hist)


u_temp = np.linspace(-1, 7, 100)
u_null = np.array([f(u_) + Eu for u_ in u_temp], dtype=np.float64)
v_null = (1 / b) * ((1.05 - Ev0) + a * u_temp)

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)

uv_hist1 = get_hist(uv0=np.array([-1, 0.43], dtype=np.float64))
plt.plot(uv_hist1[:, 0], uv_hist1[:, 1], "r")
uv_hist2 = get_hist(uv0=np.array([-1, 0.4305], dtype=np.float64))
plt.plot(uv_hist2[:, 0], uv_hist2[:, 1], "b")

plt.plot(u_temp, u_null, "k--", alpha=0.5)
plt.plot(u_temp, v_null, "r--", alpha=0.5)
plt.ylabel("v")
plt.xlabel("u")
plt.ylim([0, 2.5])
plt.xlim([-1, 7])
plt.legend(["kimochi-ii", "not-enough", "u-null", "v-null"], loc='upper right')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(dt * np.arange(L), uv_hist1[:, 0], "r")
plt.plot(dt * np.arange(L), uv_hist2[:, 0], "b")
plt.plot(dt * np.arange(L), u_B * np.ones_like(uv_hist1[:, 0]), "k--", alpha=0.5)
plt.legend(["kimochi-ii", "not-enough", "threshold"], loc='upper right')
plt.tight_layout()

plt.show()
