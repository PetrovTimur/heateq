import numpy as np
from back import solve_matrix, set_acc


class BaseSolver:
    def __init__(self, task, interval, eps):
        self.task = task
        self.interval = interval
        self.eps = eps

        print(self.interval)
        self.point_count, tau = set_acc(eps)
        self.steps = (2 * self.interval / (self.point_count - 1), tau)  # h, tau
        self.linspace = (np.linspace(-self.interval, self.interval, self.point_count))

        self.solution = self.task.alpha(self.linspace)
        self.dx, self.dt = self.steps
        self.dt = self.dx ** 2 / 4
        print(self.point_count)
        print(self.dx, self.dt)
        self.time = 0
        # print(self.linspace)

    def next(self):
        pass

    def precision_step(self):
        # dt_solution, twodt_solution = np.zeros_like(self.linspace), np.zeros_like(self.linspace)
        # while np.linalg.norm(dt_solution - twodt_solution) <= self.eps:
        #     # step with dt
        #     self.dt *= 2
        #     twodt_solution = self.next()
        #
        #     # 2 steps with 0.5 * dt
        #     self.dt /= 2
        #
        #     solution = self.solution  # temporarily save old data
        #     t = self.time
        #
        #     self.solution = self.next()  # proceed with half-step
        #     self.time += self.dt  # adjust data
        #
        #     dt_solution = self.next()
        #
        #     self.solution = solution  # return data
        #     self.time = t
        #
        #     # print(np.all(dt_solution == half_dt_solution))
        #
        #     print(self.time, np.linalg.norm(dt_solution - twodt_solution))
        #
        # # print(self.time)
        # self.dt /= 2

        dt_solution, half_dt_solution = np.ones_like(self.linspace), np.zeros_like(self.linspace)
        while np.linalg.norm(dt_solution - half_dt_solution) >= self.eps:
            # step with dt
            dt_solution = self.next()

            # 2 steps with 0.5 * dt
            self.dt /= 2

            solution = self.solution  # temporarily save old data
            t = self.time

            self.solution = self.next()  # proceed with half-step
            self.time += self.dt  # adjust data

            half_dt_solution = self.next()

            self.solution = solution  # return data
            self.time = t

            # print(np.all(dt_solution == half_dt_solution))

            print(self.time, np.linalg.norm(dt_solution - half_dt_solution))

        # print(self.time)
        self.dt *= 2
        self.time += self.dt

        # print(self.dt)

        self.solution = self.next()

        return self.solution


class SixPointSolver(BaseSolver):
    """
    Solver for: (y[i][n+1] - y[i][n]) / tau = sigma * ((y[i-1][n+1] - 2y[i][n+1] + y[i+1][n+1]) / h^2) + (1 - sigma)
    * ((y[i-1][n] - 2y[i][n] + y[i+1][n]) / h^2)
    """

    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)

        # self.sigma = 0.5 - self.steps[0] ** 2 / 12 / self.steps[1]
        self.sigma = 0.5

    def next(self):
        gamma = self.dt / self.dx ** 2

        a = [0]
        c = [1]
        b = [0]
        f = [self.task.mu1(self.time + self.dt)]

        for i in range(1, self.point_count - 1):
            a.append(gamma * self.sigma)
            c.append(1 + 2 * gamma * self.sigma)
            b.append(gamma * self.sigma)
            f.append(self.solution[i] + (1 - self.sigma) * gamma *
                     (self.solution[i - 1] - 2 * self.solution[i] + self.solution[i + 1]) +
                     self.dt * self.task.f(self.linspace[i], self.time + self.dt))

        a.append(0)
        c.append(1)
        b.append(0)
        f.append(self.task.mu2(self.time + self.dt))

        solution = np.array(solve_matrix(self.point_count - 1, a, b, c, f))

        return solution


class NonLinearSolver(BaseSolver):
    """
    Solver for: (y[i][n+1] - y[i][n]) / tau = sigma * ((y[i-1][n+1] - 2y[i][n+1] + y[i+1][n+1]) / h^2) + (1 - sigma)
    * ((y[i-1][n] - 2y[i][n] + y[i+1][n]) / h^2)
    """

    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)

    def next(self):
        # step1, half-step

        g = lambda j: 0.5 * (self.task.k(self.solution[j - 1]) + self.task.k(self.solution[j]))

        a = [0]
        c = [1]
        b = [0]
        f = [self.task.mu1(self.time + 0.5 * self.dt)]

        for i in range(1, self.point_count - 1):
            a.append(0.5 * self.dt * g(i))
            c.append(self.dx ** 2 + 0.5 * self.dt * (g(i) + g(i + 1)))
            b.append(0.5 * self.dt * g(i + 1))
            f.append(0.5 * self.dt * self.dx ** 2 * self.task.f(self.solution[i]) +
                     self.dx ** 2 * self.solution[i])

        a.append(0)
        c.append(1)
        b.append(0)
        f.append(self.task.mu2(self.time + 0.5 * self.dt))

        halfstep = np.array(solve_matrix(self.point_count - 1, a, b, c, f))

        # print(self.time, halfstep)

        # step2, full-step

        a = [0]
        c = [1]
        b = [0]
        f = [self.task.mu1(self.time + self.dt)]

        g = lambda j: 0.5 * (self.task.k(halfstep[j - 1]) + self.task.k(halfstep[j]))

        for i in range(1, self.point_count - 1):
            a.append(0.5 * self.dt * g(i))
            c.append(self.dx ** 2 + 0.5 * self.dt * (g(i + 1) + g(i)))
            b.append(0.5 * self.dt * g(i + 1))
            f.append(self.dt * self.dx ** 2 * self.task.f(halfstep[i]) + self.dx ** 2 * self.solution[i] + 0.5 *
                     self.dt * (g(i + 1) * (self.solution[i + 1] - self.solution[i]) -
                                g(i) * (self.solution[i] - self.solution[i - 1])))

        a.append(0)
        c.append(1)
        b.append(0)
        f.append(self.task.mu2(self.time + self.dt))

        solution = np.array(solve_matrix(self.point_count - 1, a, b, c, f))

        return solution


class VaryingCoefficientsSolver(BaseSolver):
    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)

    def next(self):
        sigma = 0.5

        g = lambda j: 0.5 * (self.task.k(self.linspace[j - 1], self.time + self.dt) +
                             self.task.k(self.linspace[j], self.time + self.dt))

        a = [0]
        c = [1]
        b = [0]
        f = [self.task.mu1(self.time + 0.5 * self.dt)]

        time = self.time + sigma * self.dt

        for i in range(1, self.point_count - 1):
            a.append(sigma * self.dt * g(i))
            c.append(self.task.rho(self.linspace[i], time) * self.dx ** 2 + sigma * self.dt * (g(i + 1) + g(i)))
            b.append(sigma * self.dt * g(i + 1))
            f.append(self.dt * (1 - sigma) * (g(i + 1) * (self.solution[i + 1] - self.solution[i]) -
                                              g(i) * (self.solution[i] - self.solution[i - 1])) +
                     self.dt * self.dx ** 2 * self.task.f(self.linspace[i], time) +
                     self.dx ** 2 * self.task.rho(self.linspace[i], time) * self.solution[i])

        a.append(0)
        c.append(1)
        b.append(0)
        f.append(self.task.mu2(self.time + 0.5 * self.dt))


class SemiImplicitSolver(BaseSolver):
    """Solver for"""

    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)

    def next(self):
        # step1, full-step

        a = [0]
        c = [1]
        b = [0]
        f = [self.task.mu1(self.time + self.dt)]

        g = lambda j: 0.5 * (self.task.k(self.solution[j - 1]) + self.task.k(self.solution[j]))

        for i in range(1, self.point_count - 1):
            a.append(self.dt * g(i))
            c.append(self.dx ** 2 + self.dt * (g(i + 1) + g(i)))
            b.append(self.dt * g(i + 1))
            f.append(self.dt * self.dx ** 2 * self.task.f(self.solution[i]) +
                     self.dx ** 2 * self.solution[i])

        a.append(0)
        c.append(1)
        b.append(0)
        f.append(self.task.mu2(self.time + self.dt))

        solution = np.array(solve_matrix(self.point_count - 1, a, b, c, f))

        return solution


class ImplicitSolver(BaseSolver):
    """Solver for"""

    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)
        self.M = 3

    def next(self):
        # iterate for M = 3 steps
        temp_solution = self.solution
        for s in range(self.M):
            a = [0]
            c = [1]
            b = [0]
            f = [self.task.mu1(self.time + self.dt)]

            g = lambda j: 0.5 * (self.task.k(temp_solution[j - 1]) + self.task.k(temp_solution[j]))

            for i in range(1, self.point_count - 1):
                a.append(self.dt * g(i))
                c.append(self.dx ** 2 + self.dt * (g(i + 1) + g(i)))
                b.append(self.dt * g(i + 1))
                f.append(self.dt * self.dx ** 2 * self.task.f(temp_solution[i]) +
                         self.dx ** 2 * self.solution[i])

            a.append(0)
            c.append(1)
            b.append(0)
            f.append(self.task.mu2(self.time + self.dt))

            temp_solution = np.array(solve_matrix(self.point_count - 1, a, b, c, f))

        solution = temp_solution

        return solution


class ExplicitSolver(BaseSolver):
    """Solver for"""

    def __init__(self, task, interval, eps):
        super().__init__(task, interval, eps)
        self.M = 3

    def next(self):
        #

        solution = np.empty_like(self.linspace)
        g = lambda j: 0.5 * (self.task.k(self.solution[j - 1]) + self.task.k(self.solution[j]))

        solution[0] = self.task.mu1(self.time + self.dt)
        for i in range(1, len(solution) - 1):
            solution[i] = self.solution[i] + self.dt * self.task.f(self.solution[i]) + \
                          self.dt / (self.dx ** 2) * (g(i+1) * (self.solution[i+1] - self.solution[i]) -
                                                      g(i) * (self.solution[i] - self.solution[i-1]))

        solution[-1] = self.task.mu2(self.time + self.dt)

        return solution
