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
    """Numerical integration of a function over an interval [a, b].
    
    This atom performs numerical integration using vectorized operations
    for improved performance.
    
    Parameters
    ----------
    function : callable
        Function f(x) returning a scalar or numpy array.
        The function should be vectorized (accept numpy arrays).
    a : numeric
        Lower bound of integration (must be constant)
    b : numeric  
        Upper bound of integration (must be constant)
    n : int, optional
        Number of subintervals, defaults to 1000
    method : str, optional
        Integration method, defaults to "trapezoid"
        
    Example
    -------
    # For constant integration
    result = integrate(lambda x: x**2, 0, 1)
    
    # For vectorized functions
    result = integrate(np.sin, 0, np.pi)
    """

    def __init__(self, function, a, b, n=1000, method="trapezoid") -> None:
        self.function = function
        self.n = n
        self.method = method
        self.lower_bound = float(a)
        self.upper_bound = float(b)
        
        # Create constant arguments for the parent class
        a_const = Constant(self.lower_bound)
        b_const = Constant(self.upper_bound)
        super(integrate, self).__init__(a_const, b_const)
    def validate_arguments(self) -> None:
        """Validates the arguments for the integrate atom."""
        # Validate n
        if self.n <= 0:
            raise ValueError("n must be positive")
        
        # Validate method
        valid_methods = ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"]
        if self.method not in valid_methods:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Validate Simpson's rule requirement
        if self.method == "simpsons" and self.n % 2 != 0:
            raise ValueError("For Simpson's rule, n must be even")
        
        # Integration bounds must be constants
        if not (self.args[0].is_constant() and self.args[1].is_constant()):
            raise ValueError("Integration bounds must be constants for this implementation")
        
        # Validate bounds ordering
        if (hasattr(self.args[0], 'value') and hasattr(self.args[1], 'value') and 
            self.args[0].value is not None and self.args[1].value is not None and 
            self.args[0].value > self.args[1].value):
            upper_bound = self.args[1].value
            lower_bound = self.args[0].value
            raise ValueError(f"Upper bound ({upper_bound}) must be ≥ lower bound ({lower_bound})")

        super(integrate, self).validate_arguments()
    def numeric(self, values):
        """Compute the numerical integral using vectorized operations."""
        lower_bound, upper_bound = values
        
        if lower_bound == upper_bound:
            return 0.0
            
        h = (upper_bound - lower_bound) / self.n
        
        # Generate sample points (vectorized)
        if self.method == "left_riemann":
            x_points = np.linspace(lower_bound, upper_bound - h, self.n)
            weights = np.full(self.n, h)
            
        elif self.method == "right_riemann":
            x_points = np.linspace(lower_bound + h, upper_bound, self.n)
            weights = np.full(self.n, h)
            
        elif self.method == "midpoint":
            x_points = np.linspace(lower_bound + h/2, upper_bound - h/2, self.n)
            weights = np.full(self.n, h)
            
        elif self.method == "trapezoid":
            x_points = np.linspace(lower_bound, upper_bound, self.n + 1)
            weights = np.full(self.n + 1, h)
            weights[0] = h/2
            weights[-1] = h/2
            
        elif self.method == "simpsons":
            x_points = np.linspace(lower_bound, upper_bound, self.n + 1)
            weights = np.full(self.n + 1, h/3)
            weights[0] = h/3
            weights[-1] = h/3
            weights[1::2] *= 4  # Odd indices get factor of 4
            weights[2:-1:2] *= 2  # Even indices (except endpoints) get factor of 2
        
        # Evaluate function at all points (vectorized)
        try:
            y_values = self.function(x_points)
            y_values = np.asarray(y_values)
            
            if y_values.shape == ():
                y_values = np.full_like(x_points, y_values)
                
        except (TypeError, ValueError):
            y_values = np.array([self.function(xi) for xi in x_points])
        
        # Compute integral using dot product (vectorized)
        result = np.dot(weights, y_values)
        
        if np.isscalar(result) or result.shape == ():
            return float(result)
        else:
            return result

    def _grad(self, values):
        """Gradient of the atom with respect to its arguments."""
        return [np.zeros(arg.shape) for arg in self.args]

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression."""
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression."""
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?"""
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?"""
        return False

    def get_data(self):
        """Return data needed to reconstruct the expression."""
        return [self.function, self.n, self.method]

    def graph_implementation(self, arg_objs, shape, data=None):
        """Reduces the atom to an affine expression and list of constraints."""
        values = [arg.value for arg in self.args]
        integral_value = self.numeric(values)
        return lo.create_const(integral_value, shape), []