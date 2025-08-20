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

import cvxpy as cp
from cvxpy.expressions.constants import Constant


def _substitute_params_single(expr, param_value_map):
    """
    Substitute parameter values in an expression using scalar values.
    Returns a new Expression with Parameters replaced by Constants.
    """
    if expr in param_value_map:
        return Constant(param_value_map[expr])
    if not hasattr(expr, 'args') or len(expr.args) == 0:
        return expr
    new_args = tuple(_substitute_params_single(arg, param_value_map) for arg in expr.args)
    return expr.copy(new_args)

def _create_integration_grid(parameters, a, b, n_points):
    """
    Vectorized creation of grids and trapezoidal weights for all dimensions.
    Returns:
        flattened_grids: {param: 1D numpy array of grid values}
        total_weights_flat: 1D numpy array of product of all dimension weights
    """
    dim = len(parameters)
    grids = []
    weights_1d = []
    for i in range(dim):
        xi = np.linspace(a[i], b[i], n_points)                 # grid for parameter i
        dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
        wi = np.ones(len(xi)) * dx                             # base weights
        if len(wi) > 1:
            wi[0] /= 2
            wi[-1] /= 2
        grids.append(xi)
        weights_1d.append(wi)
    # Now grids is [dim][n_points]; weights_1d likewise

    # Meshgrid makes [dim][n_points,...,n_points], one array per dim
    meshgrids = np.meshgrid(*grids, indexing='ij')             # shape: (dim, n_points^dim)
    weight_meshgrids = np.meshgrid(*weights_1d, indexing='ij') # shape: (dim, n_points^dim)

    # Vectorized weight combination:
    # total_weights = w1 * w2 * ... * wd, for each grid point
    # Do product over dim axis at each position
    total_weights = np.ones(meshgrids[0].shape)
    for w_grid in weight_meshgrids:
        total_weights *= w_grid                    # elementwise product, broadcasting

    # Flatten all for use:
    flattened_grids = {param: meshgrids[i].flatten() for i,param in enumerate(parameters)}
    total_weights_flat = total_weights.flatten()   # shape: (n_points ** dim,)

    return flattened_grids, total_weights_flat

def integrate(expression, parameters, a, b, n_points=50, method="trapezoidal"):
    """
    Numerically approximate the definite integral of a CVXPY expression 
    over one or more parameters.

    Returns a CVXPY Expression.
    """
    # Normalize parameters
    if not isinstance(parameters, (list, tuple)):
        parameters = (parameters,)
    else:
        parameters = tuple(parameters)
    dim = len(parameters)
    
    # Normalize bounds
    if not isinstance(a, (list, tuple, np.ndarray)):
        a = [a] * dim
    if not isinstance(b, (list, tuple, np.ndarray)):
        b = [b] * dim
    a = tuple(float(ai) for ai in a)
    b = tuple(float(bi) for bi in b)
    
    # Argument checks
    from cvxpy import Parameter
    for param in parameters:
        if not isinstance(param, Parameter):
            raise TypeError("All parameters must be CVXPY Parameters")
        if not param.is_scalar():
            raise ValueError("All parameters must be scalar")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if method != "trapezoidal":
        raise ValueError("Only 'trapezoidal' method supported.")

    # Build integration grid
    flattened_grids, total_weights_flat = _create_integration_grid(parameters, a, b, n_points)
    
    # Expression summation
    terms = []
    for i in range(len(total_weights_flat)):
        param_value_map = {param: float(flattened_grids[param][i]) for param in parameters}
        expr_sub = _substitute_params_single(expression, param_value_map)
        weighted_term = Constant(total_weights_flat[i]) * expr_sub
        terms.append(weighted_term)
    
    # Compose and return as CVXPY Expression
    if terms:
        integral_expr = cp.sum(cp.hstack(terms))
    else:
        integral_expr = Constant(0.0)
    return integral_expr