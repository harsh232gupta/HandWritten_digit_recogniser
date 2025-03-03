## Handwritten Digit Recognition System using Neural Networks

I developed a Handwritten Digit Recognition system using neural networks with the MNIST dataset. The dataset was split into three parts:

1. **Training dataset**
2. **Testing dataset**
3. **Cross-validation dataset**

The data was normalized using **mean normalization** and flattened into a single-dimensional array.

The neural network architecture consisted of three layers:

1. **First layer**: 120 units with ReLU activation
2. **Second layer**: 40 units with ReLU activation
3. **Output layer**: 10 units with a linear activation function

The model used the **SparseCategoricalCrossentropy** loss function with `from_logits=True` for improved numerical stability.

Various **regularization parameters** were tested to maintain a balance between high bias and high variance.
