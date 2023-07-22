import haiku as hk
import jax
import jax.numpy as jnp

import hk_multiheadattention2.MultiHeadAttention as MHA
import hk_multiheadattention2.softmax_attention as sma
import hk_multiheadattention2.fast_attention as fa

def he_init() -> hk.initializers.Initializer:
  return hk.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")


def modela(x: jax.Array) -> jax.Array:
  x = hk.MultiHeadAttention( num_heads=2
                           , key_size=int(36/2)
                           , w_init=he_init())(x,x,x)
  return x

def modelb(x: jax.Array) -> jax.Array:
  x = MHA.MultiHeadAttention( num_heads=2
                          , key_size=int(36/2)
                          , attention_fn=sma.softmax_attention_fn
                          , w_init=he_init()
                          )(x,x,x)
  return x

def modelc(x: jax.Array) -> jax.Array:
  x = MHA.MultiHeadAttention( num_heads=2
                          , key_size=int(36/2)
                          , attention_fn=fa.make_fast_softmax_attention(
                              qkv_dim = int(36/2))
                          , w_init=he_init()
                          )(x,x,x)
  return x

def modeld(x: jax.Array) -> jax.Array:
  x = MHA.MultiHeadAttention( num_heads=2
                          , key_size=int(36/2)
                          , attention_fn=fa.make_fast_generalized_attention(
                              qkv_dim = int(36/2))
                          , w_init=he_init()
                          )(x,x,x)
  return x

def main_() -> None:
  key1, key2 = jax.random.split(jax.random.PRNGKey(6546), 2)
  x = jax.random.normal(key1, (1,36,1), jnp.float32)

  print("model a")
  modela1 = hk.without_apply_rng(hk.transform(modela))
  paramsa = modela1.init(key2, x)
  ya = modela1.apply(paramsa, x)  
  # print(ya[0][0])
  
  print("model b")
  modelb1 = hk.without_apply_rng(hk.transform(modelb))
  paramsb = modelb1.init(key2, x)
  yb = modelb1.apply(paramsb, x)  
  # print(yb[0][0])
  # print(jnp.allclose(ya,yb))
  # print(jnp.array_equal(ya,yb))
  
  print("model c")
  modelc1 = hk.without_apply_rng(hk.transform(modelc))
  paramsc = modelc1.init(key2, x)
  yc = modelc1.apply(paramsc, x)  
  # print(yc[0][0])
  # # print(jnp.allclose(ya,yc))
  # # print(jnp.array_equal(ya,yc))

  print("model d")
  modeld1 = hk.without_apply_rng(hk.transform(modeld))
  paramsd = modeld1.init(key2, x)
  yd = modeld1.apply(paramsd, x)  
  # print(yd[0][0])
  # # print(jnp.allclose(ya,yd))
  # # print(jnp.array_equal(ya,yd))

main_()
