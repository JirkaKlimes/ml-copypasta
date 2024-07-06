import jax.numpy as jnp
import flax.linen as nn


class SwiGLU(nn.Module):
    """https://arxiv.org/pdf/2002.05202"""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        input_dim = x.shape[-1]

        w1 = nn.Dense(self.hidden_dim, use_bias=False)
        w2 = nn.Dense(self.hidden_dim, use_bias=False)
        w3 = nn.Dense(input_dim, use_bias=False)

        return w3(nn.silu(w1(x)) * w2(x))


if __name__ == "__main__":
    import jax

    x = jnp.ones((1, 8, 4), jnp.float32)

    layer = SwiGLU(32)
    variables = layer.init(jax.random.key(0), x)

    out = layer.apply(variables, x)

    print(out)
