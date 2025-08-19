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

import numpy as np

from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class integrate(Atom):
    """
    A CVXPY atom that numerically approximates the definite integral of a CVXPY expression 
    over one or more scalar parameters.

    This atom supports multidimensional integration over a rectangular domain, using the 
    trapezoidal rule with a user-specified number of grid points.

    Parameters
    ----------
    expression : cvxpy.Expression
        The CVXPY expression to integrate. Should be a function of the provided parameters.
    parameters : cvxpy.Parameter or iterable of cvxpy.Parameter
        Scalar CVXPY Parameters that appear in the expression and over which integration 
        will be performed.
    a : float or iterable of float
        Lower bounds of integration for each parameter. Must match the number of parameters.
    b : float or iterable of float
        Upper bounds of integration for each parameter. Must match the number of parameters.
    n_points : int, optional (default=50)
        Number of grid points per parameter dimension used in numerical integration.
        Must be at least 2.
    method : str, optional (default='trapezoidal')
        Numerical integration method. Currently only 'trapezoidal' is supported.

    Examples
    --------
    # 1D example: integrate (x + alpha)^2 over alpha in [-1, 1]
    >>> import cvxpy as cp
    >>> x = cp.Variable()
    >>> alpha = cp.Parameter()
    >>> expr = cp.square(x + alpha)
    >>> integral = integrate(expr, x, alpha, -1, 1)
    >>> problem = cp.Problem(cp.Minimize(integral))
    >>> x_opt = problem.solve()
    >>> print("Optimal x:", x.value)

    # 2D example: integrate x^2 + y^2 + (alpha*beta) over (alpha, beta) in [0,1] x [0,2]
    >>> import cvxpy as cp
    >>> x = cp.Variable()
    >>> y = cp.Variable()
    >>> alpha = cp.Parameter()
    >>> beta = cp.Parameter()
    >>> expr = cp.square(x) + cp.square(y) + alpha * beta
    >>> integral_2d = integrate(expr, [x, y], [alpha, beta], [0, 0], [1, 2])
    >>> prob = cp.Problem(cp.Minimize(integral_2d))
    >>> result = prob.solve()
    """
    def __init__(self, expression, parameters, a, b, n_points=50, method="trapezoidal"):
        self._validate_arguments(expression, parameters, a, b, n_points, method)
        
        self.expression = expression

        if not isinstance(parameters, (list, tuple)):
            parameters = (parameters,)
        else:
            parameters = tuple(parameters)
        self._parameters = parameters
        self.dim = len(self._parameters)

        if not isinstance(a, (list, tuple, np.ndarray)):
            a = [a] * self.dim
        if not isinstance(b, (list, tuple, np.ndarray)):
            b = [b] * self.dim
        self.a = tuple(float(ai) for ai in a)
        self.b = tuple(float(bi) for bi in b)

        self.n_points = int(n_points)
        self.method = method

        variables = list(expression.variables())
        self._copy_data = (expression, self._parameters, self.a, self.b, self.n_points, self.method)
        super().__init__(*variables)

    def _validate_arguments(self, expression, parameters, a, b, n_points, method):
        """Validate all input arguments for the integrate atom."""
        # Validate expression
        if not isinstance(expression, Expression):
            raise TypeError("expression must be a CVXPY Expression")
        
        # Normalize and validate parameters
        if not isinstance(parameters, (list, tuple)):
            parameters = (parameters,)
        else:
            parameters = tuple(parameters)
        
        from cvxpy import Parameter
        for param in parameters:
            if not isinstance(param, Parameter):
                raise TypeError("All parameters must be CVXPY Parameters")
            if not param.is_scalar():
                raise ValueError("All parameters must be scalar")
        
        # Validate bounds
        dim = len(parameters)
        if not isinstance(a, (list, tuple, np.ndarray)):
            a = [a] * dim
        if not isinstance(b, (list, tuple, np.ndarray)):
            b = [b] * dim
        if len(a) != dim or len(b) != dim:
            raise ValueError("Bounds must match number of parameters")
        
        a_float = tuple(float(ai) for ai in a)
        b_float = tuple(float(bi) for bi in b)
        if any(ai >= bi for ai, bi in zip(a_float, b_float)):
            raise ValueError("Each upper bound must be greater than lower bound.")
        
        # Validate n_points
        n_points_int = int(n_points)
        if n_points_int < 2:
            raise ValueError("n_points must be >= 2.")
        
        # Validate method
        if method != "trapezoidal":
            raise ValueError("Only 'trapezoidal' method supported.")

    def _substitute_params(self, expr, param_value_map):
        """Substitute parameter values in an expression."""
        if expr in param_value_map:
            return Constant(param_value_map[expr])
        if not hasattr(expr, 'args') or len(expr.args) == 0:
            return expr
        new_args = tuple(self._substitute_params(arg, param_value_map) for arg in expr.args)
        return expr.copy(new_args)

    def numeric(self, values):
        var_to_value = {var.id: values[i] for i, var in enumerate(self.variables())}
        return self._numerical_integral(var_to_value)

    def _numerical_integral(self, var_to_value):
        grids = []
        weights_1d = []
        for i in range(self.dim):
            xi = np.linspace(self.a[i], self.b[i], self.n_points)
            dx = xi[1] - xi[0]
            wi = np.ones(len(xi)) * dx
            wi[0] /= 2
            wi[-1] /= 2
            grids.append(tuple(xi))
            weights_1d.append(tuple(wi))

        total = 0.0
        meshgrids = np.meshgrid(*grids, indexing='ij')
        weight_grids = np.meshgrid(*weights_1d, indexing='ij')
        total_weights = np.ones(meshgrids[0].shape)
        for w_grid in weight_grids:
            total_weights *= w_grid

        it = np.nditer(meshgrids[0], flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            param_value_map = {}
            for j, param in enumerate(self._parameters):
                param_value_map[param] = float(meshgrids[j][idx])
            expr_sub = self._substitute_params(self.expression, param_value_map)
            old_values = {}
            for var in self.variables():
                old_values[var.id] = var.value
                var.value = var_to_value[var.id]
            try:
                expr_val = expr_sub.value
            finally:
                for var in self.variables():
                    var.value = old_values[var.id]
            total += total_weights[idx] * expr_val
            it.iternext()
        return total

    def graph_implementation(self, arg_objs, shape, data=None):
        grids = []
        weights_1d = []
        for i in range(self.dim):
            xi = np.linspace(self.a[i], self.b[i], self.n_points)
            dx = xi[1] - xi[0]
            wi = np.ones(len(xi)) * dx
            wi[0] /= 2
            wi[-1] /= 2
            grids.append(tuple(xi))
            weights_1d.append(tuple(wi))
        
        meshgrids = np.meshgrid(*grids, indexing='ij')
        weight_grids = np.meshgrid(*weights_1d, indexing='ij')
        total_weights = np.ones(meshgrids[0].shape)
        for w_grid in weight_grids:
            total_weights *= w_grid
        
        # Build sum incrementally
        total_expr = Constant(0.0)
        it = np.nditer(meshgrids[0], flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            param_value_map = {}
            for j, param in enumerate(self._parameters):
                param_value_map[param] = float(meshgrids[j][idx])
            expr_sub = self._substitute_params(self.expression, param_value_map)
            weighted_term = Constant(total_weights[idx]) * expr_sub
            total_expr = total_expr + weighted_term
            it.iternext()
        
        return total_expr.canonical_form
    
    def shape_from_args(self):
        return tuple()

    def sign_from_args(self):
        return self.expression.sign

    def is_atom_convex(self):
        return True

    def is_atom_concave(self):
        return False

    def is_incr(self, idx): return False
    def is_decr(self, idx): return False

    def is_atom_affine(self):
        return self.expression.is_affine()

    def _grad(self, values):
        return [np.zeros(arg.shape) for arg in self.args]
    
    def copy(self, args=None, id_objects=None):
        (expression, parameters, a, b, n_points, method) = self._copy_data
        return type(self)(expression, parameters, a, b, n_points, method)
    
    def is_dpp(self):
        return all(arg.is_dpp() for arg in self.args)
