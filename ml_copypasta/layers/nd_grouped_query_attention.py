import jax.numpy as jnp
import flax.linen as nn
from einops import einsum, rearrange, repeat

from ml_copypasta.layers.ndrope import NDRoPE


class NDGroupedQueryAttention(nn.Module):
    """https://arxiv.org/pdf/2305.13245"""

    q_heads: int
    kv_heads: int
    dims: int

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        q_coords: jnp.ndarray,
        kv: jnp.ndarray,
        kv_coords: jnp.ndarray,
    ):
        assert self.q_heads >= self.kv_heads, "Number of Q heads must be greater or equal to number of KV heads."
        assert self.q_heads % self.kv_heads == 0, "Number of KV heads must be divisible by number of Q heads."
        assert self.dims % self.q_heads == 0, "Number of dimensions must be divisible by number of Q heads."

        in_dims = q.shape[-1]
        head_dims = self.dims // self.q_heads
        groups = self.q_heads // self.kv_heads

        q = nn.DenseGeneral((self.q_heads, head_dims), use_bias=False)(q)
        k = nn.DenseGeneral((self.kv_heads, head_dims), use_bias=False)(kv)
        v = nn.DenseGeneral((self.kv_heads, head_dims), use_bias=False)(kv)

        q = rearrange(q, "... s (h g) d -> ... g h s d", g=groups)
        k = rearrange(k, "... s h d -> ... h s d")
        v = rearrange(v, "... s h d -> ... h s d")

        rope = NDRoPE()
        q_coords = repeat(q_coords, "... s a -> ... g h s a", g=groups, h=self.kv_heads)
        kv_coords = repeat(kv_coords, "... s a -> ... h s a", h=self.kv_heads)
        q = rope(q, q_coords)
        k = rope(k, kv_coords)

        scores = einsum(q, k, "... g h s d, ... h a d -> ... h s a")
        scale = head_dims**0.5
        attention = nn.softmax(scores / scale)

        out = einsum(attention, v, "... h s a, ... h a d -> ... h s d")
        out = rearrange(out, "... h s d -> ... s (h d)")
        out = nn.Dense(in_dims, use_bias=False)(out)
        return out


if __name__ == "__main__":
    import jax

    W, H = 8, 4
    img = jnp.ones((1, H, W, 3))
    img = rearrange(img, "b h w c -> b (h w) c")
    coords = jnp.array([(y, x) for y in range(H) for x in range(W)])[None]

    layer = NDGroupedQueryAttention(4, 2, 32)
    variables = layer.init(jax.random.key(0), img, coords, img, coords)

    out = layer.apply(variables, img, coords, img, coords)

    print(out)
