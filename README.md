# GptMini

This project showcases the development and training of two distinct language models: a simple Bigram model and a more complex GPT-inspired model. Both models are trained on a text corpus to predict the next character in a sequence resembling Shakespeare's language, demonstrating fundamental concepts in natural language processing and deep learning.

## Project Structure

- `bigram.py`: Implements a straightforward Bigram Language Model that uses token embeddings to predict the next character in a sequence.
- `gpt.py`: Features a more sophisticated model inspired by the GPT architecture, including multi-head self-attention, position-wise feed-forward networks, and layer normalization.

## Prerequisites

To run the models, ensure you have the following prerequisites installed:
- Python 3.8 or higher
- PyTorch 1.8 or higher

## Setup

1. Clone the repository to your local machine.
2. Ensure you have Python and PyTorch installed.
3. Place your training text data in a file named `input.txt` in the root directory.

## Running the Models

### Bigram Model

The Bigram Language Model (`bigram.py`) can be executed as follows:

```python bigram.py```

This script will read from input.txt, train the Bigram model, and generate text after training. The model parameters can be adjusted within the script, including batch size, block size, and learning rate.

### GPT-inspired Model

To run the GPT-inspired model (gpt.py), use the following command:

```python gpt.py```

This script also trains on input.txt and generates text post-training. Similar to the Bigram model, various hyperparameters such as batch size, block size, learning rate, and model dimensions can be customized.

## Configuration

Both scripts are configured to run with default settings suitable for demonstration purposes. For more intensive training, you may want to adjust the hyperparameters, including batch_size, block_size, learning_rate, and model dimensions (n_embd, n_head, n_layer, etc.), to fit the capacity of your hardware.

## Generating Text

After training, both models will automatically generate a sequence of text based on learned patterns. This output serves as a demonstration of the model's language understanding and generation capabilities.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
