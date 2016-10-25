"""Utilities for sympy."""
import sympy


def vector_subs(expr, substitution_vec, replace_with):
    """Substitute vectors for vectors in sympy.

    Also support numpy substitution

    >>> x = Matrix([
        a + 2,
        b * b + 1
    ])

    >>> subs_targets = Matrix([
        a,
        b
    ])

    >>> subs_replacers = Matrix([
        1.0,
        c
    ])
    >>> print vector_subs(x, subs_targets, subs_replacers)
    >>> Matrix([
        3.0,
        (c ** 2) + 1.0
    ])
    """
    sub_dict = {k: v for k, v in zip(substitution_vec, replace_with)}
    substituted_expr = [v.subs(sub_dict) for v in expr]

    return sympy.Matrix(substituted_expr)


def rk4(x, xdot, dt):
    """Symbolic RK4 integration over dt."""
    assert x.shape == xdot.shape, "x and xdot must have the same shape, wtf are you doing"

    h = sympy.Symbol('h')
    half_h = h / 2.0

    k1 = vector_subs(xdot, x, x)
    pre_k2 = vector_subs(xdot, x, x + (half_h * k1))
    k2 = sympy.simplify(pre_k2)

    pre_k3 = vector_subs(xdot, x, x + (half_h * k2))
    k3 = sympy.simplify(pre_k3)

    pre_k4 = vector_subs(xdot, x, x + (h * k3))
    k4 = sympy.simplify(pre_k4)

    x_next = x + (h / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
    return vector_subs(x_next, [h], [dt])
