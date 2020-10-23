import jax.numpy as jnp

from jax import nn
from jax.lax import rsqrt, scatter_add, ScatterDimensionNumbers


def pairwise_cosine_similarity(embeddings1, embeddings2, eps=1e-8):
    sq_norm1 = jnp.sum(embeddings1 ** 2, axis=-1, keepdims=True)
    sq_norm2 = jnp.expand_dims(jnp.sum(embeddings2 ** 2, axis=-1), axis=0)
    dot_product = jnp.matmul(embeddings1, embeddings2.transpose())
    inverse_norm = rsqrt(jnp.maximum(sq_norm1 * sq_norm2, eps ** 2))
    return dot_product * inverse_norm


def matching_log_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    num_samples = test_embeddings.shape[0]
    similarities = pairwise_cosine_similarity(embeddings, test_embeddings, eps=eps)
    logsumexp = nn.logsumexp(similarities, axis=0, keepdims=True)

    max_similarities = jnp.max(similarities, axis=0, keepdims=True)
    exp_similarities = jnp.exp(similarities - max_similarities)

    sum_exp = jnp.zeros((num_classes, num_samples), dtype=exp_similarities.dtype)
    indices = jnp.expand_dims(targets, axis=-1)
    dimension_numbers = ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    sum_exp = scatter_add(sum_exp, indices, exp_similarities, dimension_numbers)

    return jnp.log(sum_exp) + max_similarities - logsumexp


def matching_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    log_probas = matching_log_probas(embeddings,
                                     targets,
                                     test_embeddings,
                                     num_classes,
                                     eps=eps)
    return jnp.exp(log_probas)
