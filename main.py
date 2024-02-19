from plot import MyAnimation
from tasks import tasks
from solvers import SixPointSolver, NonLinearSolver, SemiImplicitSolver, ImplicitSolver, ExplicitSolver


precision = 0.01
# ani = MyAnimation(SixPointSolver, tasks[-2], precision)
# ani = MyAnimation(NonLinearSolver, tasks[-1], precision)
# ani = MyAnimation(SemiImplicitSolver, tasks[-1], precision)
ani = MyAnimation(ImplicitSolver, tasks[-1], precision)
# ani = MyAnimation(ExplicitSolver, tasks[-1], precision)
ani.animate()
