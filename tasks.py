import numpy as np


class Task:
    def __init__(self, l, f, alpha, mu1, mu2, solution=None, a=None, k=None):
        self.l = l
        self.a = a
        self.k = k
        self.f = f
        self.alpha = alpha
        self.mu1 = mu1
        self.mu2 = mu2
        self.solution = solution


task1 = Task(
    l=1,
    a=lambda x, t: 1,
    f=lambda x, t: 0,
    alpha=lambda x: np.cos(x),
    mu1=lambda t: np.exp(-t),
    mu2=lambda t: np.exp(-t) * np.cos(1),
    solution=lambda x, t: np.exp(-t) * np.cos(x)
)

task2 = Task(
    l=1,
    a=lambda x, t: 1,
    f=lambda x, t: 0,
    alpha=lambda x: np.zeros_like(x),
    mu1=lambda t: 1,
    mu2=lambda t: 1,
    solution=lambda x, t: np.exp(-t) * np.cos(x)
)

task3 = Task(
    l=1,
    a=lambda x, t: 1,
    f=lambda x, t: np.sin(x),
    alpha=lambda x: np.cos(x) + np.sin(x),
    mu1=lambda t: np.exp(-t),
    mu2=lambda t: np.exp(-t) * np.cos(1) + np.sin(1),
    solution=lambda x, t: np.exp(-t) * np.cos(x) + np.sin(x)
)

task4 = Task(
    l=1,
    a=lambda x, t: 1,
    f=lambda x, t: 2 * np.exp(t) * np.cos(x),
    alpha=lambda x: np.cos(x),
    mu1=lambda t: np.exp(t),
    mu2=lambda t: np.exp(t) * np.cos(1),
    solution=lambda x, t: np.exp(t) * np.cos(x)
)

task5 = Task(
    l=1,
    k=lambda u: 0.001 * u,
    f=lambda u: 0.003 * np.power(u, 4),
    alpha=lambda x: 5 * np.sin(np.pi * x),
    mu1=lambda t: 0,
    mu2=lambda t: 0
)

task6 = Task(
    l=1,
    k=lambda u: 0.007 * np.power(u, 2),
    f=lambda u: 0.003 * u,
    alpha=lambda x: -5 * (x - 0.2) * (x - 0.8),
    mu1=lambda t: 0,
    mu2=lambda t: 0
)

task7 = Task(
    l=1,
    k=lambda u: 0.007 * np.power(u, 2),
    f=lambda u: 0.003 * u,
    alpha=lambda x: np.abs(5.65 * np.sin(2 * np.pi * x)),
    mu1=lambda t: 0,
    mu2=lambda t: 0
)

sigma = 1
beta = sigma + 1
k0 = 1
q0 = 2
tf = 1
Lt = 2 * np.pi * np.sqrt(k0/q0) * np.sqrt((sigma + 1) / (sigma * sigma))

localisation_task = Task(
    l=Lt,
    k=lambda u: k0 * np.power(u, sigma),
    f=lambda u: q0 * np.power(u, beta),
    alpha=0,
    mu1=lambda t: 0,
    mu2=lambda t: 0,
    solution=lambda x, t: np.where(abs(x) <= Lt / 2,
                                   np.power(q0 * (tf - t), -1/sigma) *
                                   np.power(2 * (sigma + 1) / (sigma * (sigma + 2)) *
                                            np.power(np.cos(np.pi * x / Lt), 2), 1/sigma),
                                   0)
)
localisation_task.alpha = lambda x: localisation_task.solution(x, 0)

tasks = [task1, task2, task3, task4, task5, task6, task7, localisation_task]
