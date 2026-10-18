'''
Mechanistic Interpretability

The Goal:
    Is to take a trained model and reverse engineer the algorithms the model learned 
    during training from its weights.
'''

from transformer_lens import HookedTransformer


gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

number_of_layers = gpt2_small.cfg.n_layers
number_of_heads_per_layer = gpt2_small.cfg.n_heads
maximum_context_window = gpt2_small.cfg.n_ctx

print(f"Number of layers: {number_of_layers}")
print(f"Number of heads per layer: {number_of_heads_per_layer}")
print(f"Maximum context window: {maximum_context_window}")