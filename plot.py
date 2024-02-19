import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.animation as animation
import itertools


class MyAnimation:
    def __init__(self, solver, task, eps):
        self.solv = solver(task, interval=task.l, eps=eps)
        self.task = task
        self.xs = self.solv.linspace
        self.paused = False

        self.dts = []
        self.ts = []

        self.stopAt = 0
        self.fig, self.axs = plt.subplots(ncols=2, nrows=2)
        self.axs[1, 1].axis('off')

        self.axs[0, 0].grid(visible=True)
        # self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2)

        self.line1, = self.axs[0, 0].plot([], [], linestyle='None', marker='.', markersize=4)
        self.line4, = self.axs[0, 1].plot([], [])

        if self.task.solution is None:
            pass
            # self.line1, = self.axs.plot([], [], linestyle='None', marker='.')
        else:
            self.line2, = self.axs[0, 0].plot([], [])
            self.line3, = self.axs[1, 0].plot([], [])

        # self.
        self.time = self.fig.text(0.8, 0.12, 'Time: 0s')
        self.time_step = self.fig.text(0.8, 0.24, 'Step: 0.01s')

        self.ani = animation.FuncAnimation(self.fig, self.step, self.data_gen, interval=100, init_func=self.init_t,
                                           save_count=100)

    def data_gen(self):
        for cnt in itertools.count(start=0):
            yield cnt

    def init_t(self):
        y = self.solv.task.alpha
        u = self.solv.solution

        self.dts += [self.solv.dt]
        self.ts += [self.solv.time]

        self.line1.set_data(self.xs, u)
        self.line4.set_data(self.ts, self.dts)

        self.axs[0, 0].set_xlim(1.1 * np.min(self.xs), 1.1 * np.max(self.xs))
        self.axs[0, 0].set_ylim(min(u) * 0.9, max(u) * 1.1)

        self.axs[0, 1].set_xlim(1.1 * np.min(self.xs), 1.1 * np.max(self.xs))
        # self.axs[0, 1].set_ylim(0, self.solv.dt[0] * 2)
        # self.axs[0, 1].set_ylim(auto=True)
        # self.axs[0, 1].invert_yaxis()
        # self.axs[0, 1].set_yscale('log')

        if self.task.solution is None:
            pass
        else:
            self.line2.set_data(self.xs, y(self.xs))
            self.line3.set_data(self.xs, abs(u - y(self.xs)))

            self.axs[1, 0].set_xlim(1.1 * np.min(self.xs), 1.1 * np.max(self.xs))

    def step(self, data):
        t = data

        if not self.paused and (self.solv.time == 0 or (self.solv.time >= self.stopAt) and self.stopAt > 0):
            self.paused = True
            self.stopAt = -1
            self.ani.pause()
            return

        if self.paused:
            self.paused = False

        u = self.solv.precision_step()
        self.line1.set_ydata(u)
        self.time.set_text(f'Time: {self.solv.time:.8f}s')
        self.time_step.set_text(f'Step: {self.solv.dt:.8f}s')

        self.dts += [self.solv.dt]
        self.ts += [self.solv.time]
        self.line4.set_data(self.ts, self.dts)
        self.axs[0, 1].set_xlim(-0.1, self.ts[-1] + 1)
        self.axs[0, 1].set_ylim(self.line4.get_data()[1][-1] / 2, self.line4.get_data()[1][0] + self.line4.get_data()[1][-1])
        # self.axs[0, 1].invert_yaxis()
        # self.axs[0, 1].set_ylim(auto=True)
        # self.axs[0, 1].set_yscale('log')

        self.axs[0, 0].set_ylim(min(self.solv.solution) * 0.9, max(self.solv.solution) * 1.1)

        if self.task.solution is None:
            return self.line1, self.time,
        else:
            # print(self.solv.solution)
            y = self.task.solution(self.xs, self.solv.time)
            self.line2.set_ydata(y)
            self.line3.set_data(self.xs, abs(u - y))

            self.axs[1, 0].set_ylim(min(abs(u - y)) * 0.9, max(abs(u - y)) * 1.1)

            return self.line1, self.line2, self.line3, self.time,

    def setStopTime(self, val):
        print(f'Stop time set at: {float(val)}s')
        self.stopAt = float(val)

    def animate(self):
        self.axs[0, 0].set_xlabel('Coord [x]')
        self.axs[0, 0].set_ylabel('Val [u]')
        self.axs[0, 0].legend(['Numerical', 'Analytical'], loc='best')

        if self.task.solution is None:
            pass
        else:
            self.axs[0, 1].set_xlabel('Coord [x]')
            self.axs[0, 1].set_ylabel('Val [u]')
            self.axs[0, 1].legend(['Numerical', 'Analytical'], loc='best')
            pass

        axplay = self.fig.add_axes([0.55, 0.3, 0.10, 0.05])
        bplay = Button(axplay, 'Play')
        axpause = self.fig.add_axes([0.55, 0.2, 0.10, 0.05])
        bpause = Button(axpause, 'Pause')

        axbox = self.fig.add_axes([0.55, 0.1, 0.1, 0.05])
        tbox = TextBox(axbox, label="Stop at: ")
        tbox.on_submit(self.setStopTime)

        bplay.on_clicked(lambda event: self.ani.resume())
        bpause.on_clicked(lambda event: self.ani.pause())

        plt.show()
