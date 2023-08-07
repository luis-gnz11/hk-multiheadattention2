# hk-multiheadattention2
A variation of the MultiHeadAttention for the Deepmind's Haiku framework that allows for other attention functions.
The package include implementions for the following attention functions:
  - softmax
  - FAVOR+ (https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py)
  
# local usage
- clone locally
- using poetry:
  - poetry add /path/to/local/clone/hk-multiheadattention2/
- or using pip:
  - cd /path/to/local/clone/hk-multiheadattention2/
  - pip install -e .
  - 
