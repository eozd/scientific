"""
This code is adapted from https://github.com/titu1994/tfdiffeq/blob/master/tfdiffeq/adjoint.py
but instead of using the dopri5 solver provided with tensorflow scientific, it uses BDF algorithm from
tensorflow probability, which is supposed to be numerically more stable in stiff regions than dopri5.
"""

import numpy as np
import tensorflow as tf
from tensorflow_scientific.integrate import odeint
from tensorflow_probability.python.math.ode import BDF
from tensorflow_scientific.integrate.utils import flatten
from collections import namedtuple


_Arguments = namedtuple('_Arguments', 'func bdf_solver')
_arguments = None

@tf.custom_gradient
def adjoint_method_bdf(y0, t):
    global _arguments
    func = _arguments.func
    bdf_solver = _arguments.bdf_solver

    # Forward integration
    # ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
    print('Calling the BDF solver')
    ans = bdf_solver.solve(func, t[0], y0, t).states

    def grad_fn(*grad_output, **kwargs):
        variables = kwargs.get('variables', None)
        global _arguments
        f_params = tuple(variables)
        flat_params = flatten(variables)
        func = _arguments.func
        bdf_solver = _arguments.bdf_solver
        n_states = ans.shape[0] if len(ans.shape) == 1 else ans.shape[1]

        def augmented_dynamics(t, y_aug):
            tpos = -t
            y, adj_y = y_aug[:n_states], y_aug[n_states:2 * n_states]

            with tf.GradientTape() as tape:
                tape.watch(tpos)
                tape.watch(y)
                tape.watch(f_params)
                func_eval = func(tpos, y)

            gradys = -adj_y
            if len(gradys.shape) < len(func_eval.shape):
                gradys = tf.expand_dims(gradys, axis=0)

            vjp_t, vjp_y, vjp_params = tape.gradient(
                func_eval,
                (t, y) + f_params,
                output_gradients=gradys,
                unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            return -flatten((func_eval, vjp_y, vjp_t, vjp_params))

        # Backward integration using augmented state
        T = ans.shape[0]
        adj_y = grad_output[-1][-1]
        adj_params = tf.zeros_like(flat_params, dtype=flat_params.dtype)
        adj_time = tf.convert_to_tensor(0., dtype=t.dtype)
        time_vjps = []
        for i in range(T - 1, 0, -1):
            func_i = func(t[i], ans[i])
            grad_output_i = grad_output[-1][i]

            # Compute the effect of moving the current time measurement point.
            dLd_cur_t = tf.tensordot(tf.squeeze(func_i), tf.squeeze(grad_output_i), 1)
            adj_time = adj_time - dLd_cur_t
            time_vjps.append(dLd_cur_t)

            aug_y0 = flatten((ans[i], adj_y, adj_time, adj_params))

            aug_ans = bdf_solver.solve(
                augmented_dynamics,
                -t[i],
                aug_y0,
                tf.convert_to_tensor([-t[i], -t[i - 1]]),
                # rtol=rtol, atol=atol, method=method, options=options
            ).states

            # Unpack aug_ans.
            adj_y = aug_ans[1, n_states:2 * n_states]
            adj_time = aug_ans[1, 2 * n_states]
            adj_params = aug_ans[1, 2 * n_states + 1:]

            adj_y += grad_output[-1][i-1]

        time_vjps.append(adj_time)
        time_vjps = flatten(time_vjps)

        adj_params_list = []
        beg = 0
        for v in variables:
            shape = v.shape
            size = tf.size(v)
            end = beg + size
            adj_params_list.append(tf.reshape(adj_params[beg:end], shape))
            beg = end

        return (adj_y, time_vjps), adj_params_list

    return ans, grad_fn


def odeint_adjoint_bdf(func, y0, t, bdf_solver):
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')
    if not func.built:
        _ = func(t, y0)
    global _arguments
    _arguments = _Arguments(func, bdf_solver)
    return adjoint_method_bdf(y0, t)
