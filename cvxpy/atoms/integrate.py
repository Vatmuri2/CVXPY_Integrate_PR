"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple

import numpy as np

import cvxpy as cp
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression


class integrate(Atom):
    """Numerical integration of a function over an interval [a, b].
    
    Supports Riemann sums (left/right/midpoint), trapezoid rule, and Simpson's rule.
    The function must return CVXPY expressions or constants.
    
    Parameters
    ----------
    function : callable
        Function f(x) returning a scalar or CVXPY expression
    a : numeric or CVXPY expression
        Lower bound of integration
    b : numeric or CVXPY expression
        Upper bound of integration
    n : int, optional
        Number of subintervals, defaults to 1000 (must be even for Simpson's rule)
    method : str, optional
        Integration method, defaults to "trapezoid". 
        One of ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"]
        
    Example
    -------
    x = cp.Variable()
    expr = integrate(lambda t: cp.exp(x*t), 0, 1)
    """
    def __init__(self, function, a, b, n=1000, method="trapezoid"):
        self.function = function
        self.a = a if isinstance(a, Expression) else cp.Constant(a)
        self.b = b if isinstance(b, Expression) else cp.Constant(b)
        self.n = n
        self.method = method
        
        # Validate inputs
        if n <= 0:
            raise ValueError("n must be positive")
        if method not in ["left_riemann", "right_riemann", "midpoint", "trapezoid", "simpsons"]:
            raise ValueError(f"Unsupported method: {method}")
        if method == "simpsons" and n % 2 != 0:
            self.n = n + 1  # Force even n for Simpson's rule
            
        # Test evaluation to determine output shape
        has_a_value = hasattr(self.a, 'value')
        has_b_value = hasattr(self.b, 'value')
        if has_a_value and has_b_value:
            test_x = (self.a.value + self.b.value) / 2
        else:
            test_x = 0.5        
        test_expr = self.function(test_x)
        if isinstance(test_expr, Expression):
            self._shape = test_expr.shape
        else:
            self._shape = ()  # Scalar output
        
        super().__init__(self.a, self.b)

    def numeric(self, values) -> np.ndarray:
        """Numerical integration implementation."""
        a_val, b_val = values
        
        # Early return for zero-width interval
        if a_val == b_val:
            if not self._shape:  # Scalar output
                return 0
            else:
                return np.zeros(self._shape)
            
        h = (b_val - a_val) / self.n
        
        # Generate sample points
        if self.method == "left_riemann":
            x = np.linspace(a_val, b_val - h, self.n)
        elif self.method == "right_riemann":
            x = np.linspace(a_val + h, b_val, self.n)
        elif self.method == "midpoint":
            x = np.linspace(a_val + h/2, b_val - h/2, self.n)
        elif self.method == "trapezoid":
            x = np.linspace(a_val, b_val, self.n + 1)
        elif self.method == "simpsons":
            x = np.linspace(a_val, b_val, self.n + 1)
        
        # Evaluate function at all points
        y_values = []
        for xi in x:
            y = self.function(xi)
            if isinstance(y, Expression) and hasattr(y, 'value'):
                y_values.append(y.value)
            else:
                y_values.append(y)
        y = np.array(y_values)
        
        # Compute integral
        if self.method in ["left_riemann", "right_riemann", "midpoint"]:
            return h * np.sum(y, axis=0)
        elif self.method == "trapezoid":
            s = np.sum(y[1:-1], axis=0) if self.n > 1 else 0
            return h * (0.5 * (y[0] + y[-1]) + s)
        elif self.method == "simpsons":
            odd_sum = np.sum(y[1:-1:2], axis=0)
            even_sum = np.sum(y[2:-2:2], axis=0) if self.n > 2 else 0
            return (h / 3) * (y[0] + y[-1] + 4 * odd_sum + 2 * even_sum)

    def validate_arguments(self) -> None:
        """Check that a < b and function returns valid expressions."""
        if not (self.a.is_constant() and self.b.is_constant()):
            raise ValueError("Integration bounds a and b must be constants")
        if hasattr(self.a, 'value') and hasattr(self.b, 'value') and self.a.value > self.b.value:
            raise ValueError(f"b ({self.b.value}) must be ≥ a ({self.a.value})")

    def shape_from_args(self):
        """Return the shape of the integral output."""
        return self._shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression."""
        # Generally unknown without knowledge of the function
        return (False, False)

    def get_data(self) -> List:
        """Return data needed to reconstruct the expression."""
        return [self.function, self.n, self.method]

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        # Depends on the function being integrated
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        # Depends on the function being integrated
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?"""
        # Generally unknown without knowledge of the function
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?"""
        # Generally unknown without knowledge of the function
        return False
        
    def _grad(self, values):
        """Gradient of the atom with respect to its arguments.
        
        This is a required method for all Atom subclasses.
        """
        # Since we're just doing numerical integration, we don't have
        # a closed-form gradient. The DCP rules will handle this.
        return None