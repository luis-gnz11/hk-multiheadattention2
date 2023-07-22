import jax
from typing import Optional
import jax.numpy as jnp

def softmax_attention_fn(
    query_heads: jax.Array,
    key_heads: jax.Array,
    value_heads: jax.Array,
    mask: Optional[jax.Array] = None
) -> jax.Array:
  # Compute attention weights.
  *_, key_size = key_heads.shape
  attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
  attn_logits = attn_logits / jnp.sqrt(key_size).astype(key_heads.dtype)
  if mask is not None:
    if mask.ndim != attn_logits.ndim:
      raise ValueError(
          f"Mask dimensionality {mask.ndim} must match logits dimensionality "
          f"{attn_logits.ndim}."
      )
    attn_logits = jnp.where(mask, attn_logits, -1e30)
  attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]
  attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)  # weight the values by the attention
  return attn
  