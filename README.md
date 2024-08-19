# ARMA: Mitigating Catastrophic Forgetting using Attention-Regularized Model Averaging in Continual Fine-tuning Large Language Models

## Introduction

Welcome to the repository for **ARMA**, a novel framework designed to address catastrophic forgetting in large language models (LLMs) during continual fine-tuning. Catastrophic forgetting is a common challenge when fine-tuning models on domain-specific tasks, often leading to a decline in performance on previously learned tasks. ARMA mitigates this issue by utilizing attention-regularized model averaging, which balances the trade-off between domain-specific improvements and general task performance.


```markdown
# model_merge_optimizer

## Introduction

Welcome to the repository for **model_merge_optimizer**, a Python script designed to merge two causal language models by optimizing attention layers' weights. This approach allows for a more nuanced blending of model capabilities, potentially improving performance on specific tasks by fine-tuning the balance between two models.

## Features

- **Model Merging with Attention Optimization**: The script merges two pre-trained causal language models by adjusting their attention layers' weights, allowing for fine-tuned control over model performance.
- **Customizable Alpha Blending**: Provides an alpha parameter to control the blending ratio between the two models, which is optimized using gradient descent.
- **Multiple Difference Metrics**: Includes functions to compute differences between models using MSE, cosine similarity, Earth Mover's Distance (EMD), and KL divergence.

## Installation

To use this script, you need to have Python 3.7 or later installed, along with the following libraries:

```bash
pip install torch transformers
```

## Usage

### Loading Models

You can load your models using the `AutoModelForCausalLM` and `AutoTokenizer` from the Hugging Face `transformers` library. Replace the `checkpoint_1` and `checkpoint_2` paths with the paths to your models.

### Merging Models

The `merge_models` function allows you to merge the two models using a specified alpha value. This function combines the attention layers' weights from both models based on the alpha parameter.

### Optimizing Alpha

The `find_optimal_alpha` function automates the process of finding the optimal alpha by minimizing the difference between the merged model and one of the original models using gradient descent.


## Example

```python
# Merge two models with an initial alpha of 0.5
alpha = find_optimal_alpha(model_1, model_2, data_loader, initial_alpha=0.5)
merged_model = merge_models(model_1, model_2, alpha)
```


## Contributing

Contributions are welcome! Please submit pull requests or open issues to contribute to the project.

## License

This project is licensed under the MIT License. Please refer to the `LICENSE.md` file for more details.

## Acknowledgments



For any issues or questions, please open an issue in this repository or contact the maintainers directly.

Happy coding!
```

This `README.md` reflects the specific functionality of the code provided and should help users understand how to implement and optimize the model merging process.
