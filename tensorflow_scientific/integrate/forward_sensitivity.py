import numpy as np
import tensorflow as tf
from tensorflow_scientific.integrate import odeint
from tensorflow_scientific.integrate.utils import flatten
from collections import namedtuple


_Arguments = namedtuple('_Arguments', 'func method options rtol atol, t')
_arguments = None

@tf.custom_gradient
def forward_sensitivity_method(y0):
    # Code adapted to TF from https://docs.pymc.io/notebooks/ODE_with_manual_gradients.html
    expanded = len(y0.shape) > 1

    global _arguments
    func = _arguments.func
    method = _arguments.method
    options = _arguments.options
    rtol = _arguments.rtol
    atol = _arguments.atol
    t = _arguments.t
    y0 = tf.squeeze(y0)
    n_states = tf.squeeze(y0).shape[0]
    n_theta = tf.squeeze(func.theta, -1).shape[0]
    n_ivs = n_states

    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
    T = ans.shape[0]

    def grad_fn(*grad_output, **kwargs):
        variables = kwargs.get('variables', None)
        f_params = tuple(tf.squeeze(v, -1) for v in variables)

        # Augmented forward integration
        def dfdx(x, t):
            with tf.GradientTape() as tape:
                tape.watch(x)
                func_out = func(x, t)
            return tape.jacobian(func_out, x, unconnected_gradients='zero')

        def dfdp(x, t):
            with tf.GradientTape() as tape:
                tape.watch(f_params)
                func_out = func(x, t)
            jac_list = tape.jacobian(func_out, f_params + (y0,), unconnected_gradients='zero')
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
        dydp = tf.reshape(result[:, n_states:], [T, n_states, n_theta + n_ivs])
        dydtheta = dydp[:, :, :n_theta]
        dydy0 = dydp[:, :, n_theta:]

        def vec_jac_prod(dydp, dLdy):
            dydp_T = tf.transpose(dydp, [0, 2, 1])
            if len(dLdy.shape) < len(dydp_T.shape):
                dLdy = tf.expand_dims(dLdy, -1)
            return tf.squeeze(tf.math.reduce_sum(tf.matmul(dydp_T, dLdy), axis=0), -1)

        grad_output = grad_output[-1]
        dLdtheta = vec_jac_prod(dydtheta, grad_output)
        dLdy0 = vec_jac_prod(dydy0, grad_output)

        dLdtheta_list = []
        beg = 0
        for v in variables:
            shape = v.shape
            size = tf.size(v)
            end = beg + size
            dLdtheta_list.append(tf.reshape(dLdtheta[beg:end], shape))
            beg = end

        return dLdy0, dLdtheta_list

    if expanded:
        ans = tf.expand_dims(ans, -1)
    return ans, grad_fn


def odeint_forward_sensitivity(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None):
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')
    if not func.built:
        _ = func(y0, t)
    global _arguments
    _arguments = _Arguments(func, method, options, rtol, atol, t)
    return forward_sensitivity_method(y0)
