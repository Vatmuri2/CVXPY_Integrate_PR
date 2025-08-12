"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant


class integrate(Atom):
    """Numerical integration of a function over a (possibly multidimensional) box [a, b].

    Supports multi-D integration if `a` and `b` are sequences (lists or arrays).
    Supported methods: "left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons".
    Simpson's rule requires even intervals per dimension.

    Parameters
    ----------
    function : callable
        Function f(z;x) returning a scalar, where z is a set of cvxpy
        variables and x is a list of cvxp parameters, and f(z;x) returns a valid
        convex cxvpy expression. 
        Should support vectorized evaluation if possible.
    a : scalar or list
        Lower bound(s) (must be constant). 
    b : scalar or list
        Upper bound(s) (must be constant).
    n : int or list, optional
        Number of subintervals per dimension, defaults to 1000.
    method : str, optional
        Integration method; default is "trapezoid".

    Example
    -------

    # 1D
    a = cvx.Variable()
    b = cvx.Variable(pos=True)
    x = cvx.Parameter()

    f = lambda x: cvx.square(a * x + b)
    obj = cvx.integrate(f, x, 0, 1, 200)

    # Multi-D
    a = cvx.Variable()
    b = cvx.Variable(pos=True)
    x = cvx.Parameter(2)
    
    f = lambda c: cvx.square(a * x[0] + b * x[1])
    obj = cvx.integrate(f, x, [0,0], [1,1], [50, 100], method="simpsons")


    Notes:
       z = cvx.Variable(2)
       x = cvx.Parameter(2)

       def g(z, x):
           return z[0]*x[0] + z[1] * x[1]

       obj = cvx.integrate(g, z, x, [0,1], [1,1], [20, 50])

       prob = cvx.Problem(g, [z[0]>=0, z[0] + z[1] >= -2])
       
    ## within cvx.integrate

        x_0 = np.linspace(lower[0], upper[0], n[0])
        x_1 = np.linspace(lower[1], upper[1], n[1])
        ...

        [X0,X1] = np.meshgrid(x_0, x_1)

        gij_lambda = lambda x: g(z,x)

        integral = 0
        for i in range(n[0]):
            for j in range(n[1]):
                integral += weight[i,j] *  gij_lambda([X0[i,j], X1[i,j]])

        return cvx.sum(integral)

    """

    def __init__(self, function, a, b, n=1000, method="trapezoid") -> None:
        self.function = function
        self.method = method

        a_arr = np.atleast_1d(a)
        b_arr = np.atleast_1d(b)
        if a_arr.shape != b_arr.shape:
            raise ValueError(f"Bounds a{a_arr.shape} and b{b_arr.shape} must match shapes.")
        self.dim = a_arr.size

        if isinstance(n, int):
            self.n = [n] * self.dim
        else:
            self.n = list(n)
            if len(self.n) != self.dim:
                msg = f"n must match number of dimensions: got n={self.n}, dim={self.dim}"
                raise ValueError(msg)

        self.lower_bounds = a_arr.astype(float)
        self.upper_bounds = b_arr.astype(float)

        a_const = Constant(self.lower_bounds)
        b_const = Constant(self.upper_bounds)
        super().__init__(a_const, b_const)

    def validate_arguments(self) -> None:
        valid_methods = ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"]
        if self.method not in valid_methods:
            raise ValueError(f"Unsupported method: {self.method}")
        if any(ni <= 0 for ni in self.n):
            raise ValueError("All n (subintervals per dimension) must be positive.")
        if self.method == "simpsons":
            for ax, ni in enumerate(self.n):
                if ni % 2 != 0:
                    msg = f"For simpsons method, n on axis {ax} must be even (got {ni})"
                    raise ValueError(msg)
        if not (self.args[0].is_constant() and self.args[1].is_constant()):
            raise ValueError("Integration bounds must be constants for this implementation.")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("Upper bounds must be ≥ lower bounds.")
        super().validate_arguments()

    def _get_grid_and_weights(self, a, b):
        grids = []
        weights_axes = []
        for axis in range(self.dim):
            ni = self.n[axis]
            ai, bi = a[axis], b[axis]
            h = (bi - ai) / ni

            if self.method == "left_riemann":
                pts = np.linspace(ai, bi - h, ni)
                ws = np.full(ni, h)
            elif self.method == "right_riemann":
                pts = np.linspace(ai + h, bi, ni)
                ws = np.full(ni, h)
            elif self.method == "midpoint":
                pts = np.linspace(ai + h / 2, bi - h / 2, ni)
                ws = np.full(ni, h)
            elif self.method == "trapezoid":
                pts = np.linspace(ai, bi, ni + 1)
                ws = np.full(ni + 1, h)
                ws[0] = h / 2
                ws[-1] = h / 2
            elif self.method == "simpsons":
                # Simpson's rule: n must be even
                pts = np.linspace(ai, bi, ni + 1)
                ws = np.full(ni + 1, h / 3)
                ws[0] = h / 3
                ws[-1] = h / 3
                ws[1:-1:2] *= 4    # odd indices
                ws[2:-1:2] *= 2    # even indices (not ends)
            grids.append(pts)
            weights_axes.append(ws)
        return grids, weights_axes

    def numeric(self, values):
        "Numerical integration: ND tensor product cubature over rectangular box, supports simpsons."
        a = np.atleast_1d(values[0])
        b = np.atleast_1d(values[1])
        if np.all(a == b):
            return 0.0

        grids, weights_axes = self._get_grid_and_weights(a, b)
        mesh = np.meshgrid(*grids, indexing='ij')
        mesh_weights = np.meshgrid(*weights_axes, indexing='ij')
        # ND grid: shape (num_points, dim)
        points = np.stack([m.ravel() for m in mesh], axis=-1)
        weights = np.prod(np.stack(mesh_weights, axis=-1), axis=-1).ravel()

        # Evaluate function; attempt vectorized first
        try:
            vals = self.function(*[arr.ravel() for arr in mesh])
        except Exception:
            vals = np.array([self.function(*pt) for pt in points])

        vals = np.asarray(vals)
        if vals.shape == ():
            vals = np.full(points.shape[0], vals)

        result = np.dot(weights, vals)
        return float(result) if np.isscalar(result) or vals.shape == () else result

    def _grad(self, values):
        return [np.zeros(arg.shape) for arg in self.args]

    def shape_from_args(self) -> Tuple[int, ...]:
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_atom_convex(self) -> bool:
        return True

    def is_atom_concave(self) -> bool:
        return False

    def is_incr(self, idx) -> bool:
        return False

    def is_decr(self, idx) -> bool:
        return False

    def get_data(self):
        return [self.function, self.n, self.method]

    def graph_implementation(self, arg_objs, shape, data=None):
        values = [arg.value for arg in self.args]
        integral_value = self.numeric(values)
        return lo.create_const(integral_value, shape), []
