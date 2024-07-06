import jax.numpy as jnp
import flax.linen as nn


class NDRoPE(nn.Module):
    """https://arxiv.org/pdf/2403.13298v1"""

    @nn.compact
    def __call__(self, x: jnp.ndarray, coordinates: jnp.ndarray):
        dims = x.shape[-1]
        assert dims & ~1, "Number of input dimensions must be even."

        angles = nn.Dense(dims // 2, use_bias=False)(coordinates)
        angles = jnp.e ** (1j * angles)

        x_complex = x[..., ::2] + 1j * x[..., 1::2]
        x_rotated = x_complex * angles

        x_out = jnp.empty_like(x)
        x_out = x_out.at[..., ::2].set(x_rotated.real)
        x_out = x_out.at[..., 1::2].set(x_rotated.imag)

        return x_out


if __name__ == "__main__":
    import jax

    x = jnp.ones((1, 8, 4), jnp.float32)
    coordinates = jnp.arange(8).reshape(1, 8, 1)

    layer = NDRoPE()
    variables = layer.init(jax.random.key(0), x, coordinates)

    out = layer.apply(variables, x, coordinates)

    print(out)
