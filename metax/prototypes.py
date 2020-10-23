import jax.numpy as jnp

from jax.lax import scatter_add, ScatterDimensionNumbers


def get_num_samples(targets, num_classes, dtype=None):
    ones = jnp.ones_like(targets, dtype=dtype)
    indices = jnp.expand_dims(targets, axis=-1)
    num_samples = jnp.zeros(targets.shape[:-1] + (num_classes,), dtype=ones.dtype)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    return scatter_add(num_samples, indices, ones, dimension_numbers)


def get_prototypes(embeddings, targets, num_classes):
    embedding_size, dtype = embeddings.shape[-1], embeddings.dtype
    num_samples = get_num_samples(targets, num_classes, dtype=dtype)
    num_samples = jnp.expand_dims(jnp.maximum(num_samples, 1), axis=-1)

    prototypes = jnp.zeros((num_classes, embedding_size), dtype=dtype)
    indices = jnp.expand_dims(targets, axis=-1)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    prototypes = scatter_add(prototypes, indices, embeddings, dimension_numbers)

    return prototypes / num_samples
