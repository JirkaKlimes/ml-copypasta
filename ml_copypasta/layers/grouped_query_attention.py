import jax.numpy as jnp
import flax.linen as nn
from einops import einsum, rearrange, reduce

from ml_copypasta.layers.rope import RoPE


class GroupedQueryAttention(nn.Module):
    """https://arxiv.org/pdf/2305.13245"""

    q_heads: int
    kv_heads: int
    dims: int

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        kv: jnp.ndarray,
    ):
        assert (
            self.q_heads >= self.kv_heads
        ), "Number of Q heads must be greater or equal to number of KV heads."
        assert (
            self.q_heads % self.kv_heads == 0
        ), "Number of KV heads must be divisible by number of Q heads."
        assert (
            self.dims % self.q_heads == 0
        ), "Number of dimensions must be divisible by number of Q heads."

        in_dim = q.shape[-1]
        head_dim = self.dims // self.q_heads
        groups = self.q_heads // self.kv_heads

        q = nn.DenseGeneral((self.q_heads, head_dim), use_bias=False)(q)
        k = nn.DenseGeneral((self.kv_heads, head_dim), use_bias=False)(kv)
        v = nn.DenseGeneral((self.kv_heads, head_dim), use_bias=False)(kv)

        q = rearrange(q, "... s (h g) d -> ... g h s d", g=groups)
        k = rearrange(k, "... s h d -> ... h s d")
        v = rearrange(v, "... s h d -> ... h s d")

        q = RoPE()(q)
        k = RoPE()(k)

        scores = einsum(q, k, "... g h s d, ... h a d -> ... h s a")
        scale = head_dim**0.5
        attention = nn.softmax(scores / scale)

        out = einsum(attention, v, "... h s a, ... h a d -> ... h s d")
        out = rearrange(out, "... h s d -> ... s h d")
        out = reduce(out, "... s h d -> ... s d", "mean")
        out = nn.Dense(in_dim, use_bias=False)(out)
        return out


if __name__ == "__main__":
    import jax

    seq = jnp.ones((1, 8, 4))

    layer = GroupedQueryAttention(4, 2, 32)
    variables = layer.init(jax.random.key(0), seq, seq)

    out = layer.apply(variables, seq, seq)

    print(out)
