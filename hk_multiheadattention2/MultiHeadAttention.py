from typing import Optional
import warnings

from haiku._src import basic
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp
import haiku as hk

class MultiHeadAttention(hk.Module):
  def __init__(
      self,
      num_heads: int,
      key_size: int,
      attention_fn,
      w_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      b_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    self.w_init = w_init
    self.with_bias = with_bias
    self.b_init = b_init
    self.attention_fn = attention_fn

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
    key_heads = projection(key, self.key_size, "key")  # [T, H, K]
    value_heads = projection(value, self.value_size, "value")  # [T, H, V]

    attn = self.attention_fn(query_heads, key_heads, value_heads, mask=mask)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  #  flatten the head vectors: [T', H*V] 

    print(f"A.shape = {attn.shape}")
    print(f"A = {attn}")

    # Apply another projection to get the final embeddings.
    final_projection = hk.Linear(self.model_size, w_init=self.w_init,
                                 with_bias=self.with_bias, b_init=self.b_init)
    return final_projection(attn)  # [T', D']

  @hk.transparent
  def _linear_projection(
      self,
      x: jax.Array,
      head_size: int,
      name: Optional[str] = None,
  ) -> jax.Array:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init,
                  with_bias=self.with_bias, b_init=self.b_init, name=name)(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))

