import numpy as np
import tensorflow as tf
from tensorflow_scientific.integrate import odeint
from tensorflow_scientific.integrate.utils import flatten
from collections import namedtuple


_Arguments = namedtuple('_Arguments', 'func return_sensitivities method options rtol atol')
_arguments = None

def forward_sensitivity_method(y0, t):
    # Code adapted to TF from https://docs.pymc.io/notebooks/ODE_with_manual_gradients.html

    global _arguments
    func = _arguments.func
    method = _arguments.method
    options = _arguments.options
    rtol = _arguments.rtol
    atol = _arguments.atol
    return_sensitivities = _arguments.return_sensitivities
    n_states = len(tf.squeeze(y0))
    n_theta = len(tf.squeeze(func.theta))
    n_ivs = n_states

    if return_sensitivities:
        # Augmented forward integration
        def dfdx(x, t):
            with tf.GradientTape() as tape:
                tape.watch(x)
                func_out = func(x, t)
            return tape.jacobian(func_out, x, unconnected_gradients='zero')

        def dfdp(x, t):
            with tf.GradientTape() as tape:
                tape.watch(func.theta)
                tape.watch(y0)
                func_out = func(x, t)
            jac_list = tape.jacobian(func_out, [func.theta, y0], unconnected_gradients='zero')
            return tf.concat(jac_list, axis=1)

        def aug_func(x_aug, t):
            x = x_aug[:n_states]
            dxdp = tf.reshape(x_aug[n_states:], [n_states, n_theta + n_ivs])

            dxdt = func(x, t)
            d_dxdp_dt = tf.matmul(dfdx(x, t), dxdp) + dfdp(x, t)
            return flatten([dxdt, d_dxdp_dt])
        
        aug_rest = np.zeros(n_states * (n_theta + n_ivs))
        for i in range(n_ivs):
            offset = n_theta * (i + 1) + n_ivs * i + i
            aug_rest[offset] = 1.0
        y0_aug = tf.cast(flatten([y0, aug_rest]), tf.float64)

        result = odeint(aug_func, y0_aug, t, rtol=rtol, atol=atol, method=method, options=options)
        y = result[:, :n_states]
        dydp = tf.reshape(result[:, n_states:], [len(t), n_states, n_theta + n_ivs])
        dydtheta = dydp[:, :, :n_theta]
        dydy0 = dydp[:, :, n_theta:]
        ans = (y, dydtheta, dydy0)
    else:
        ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    return ans


def odeint_forward_sensitivity(func, y0, t, return_sensitivities=False, rtol=1e-6, atol=1e-12, method=None, options=None):
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')
    if not func.built:
        _ = func(y0, t)
    global _arguments
    _arguments = _Arguments(func, return_sensitivities, method, options, rtol, atol)
    return forward_sensitivity_method(y0, t)
