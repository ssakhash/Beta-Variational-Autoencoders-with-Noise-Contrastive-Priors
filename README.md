# Beta-Variational-Autoencoders-with-Noise-Contrastive-Priors

## Overview
This project implements a Beta-Variational Autoencoder (Beta-VAE) with Noise Contrastive Priors. Beta-VAEs are a type of Variational Autoencoder (VAE) that constrains the capacity of the latent space using an additional hyperparameter, β. This encourages the model to learn more interpretable, disentangled representations. The Noise Contrastive Priors (NCP) technique further enhances this project by introducing an energy-based binary classifier trained to distinguish between the posterior samples and noise samples in the latent space.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- torchvision

## Installation
```
pip install torch torchvision numpy matplotlib
```

## Usage
To use this project, simply clone the repository and run the provided Python script using Jupyter Notebook or Google Colab:
```
git clone https://github.com/ssakhash/Beta-Variational-Autoencoders-with-Noise-Contrastive-Priors.git
```

## Architecture
The project consists of the following main components:

1) Variational Autoencoder (VAE): A VAE is a probabilistic generative model that is trained to encode and decode data in a lower-dimensional latent space. In this project, it is implemented as a PyTorch module, with separate encoder and decoder neural networks.

2) Beta Regularization: In addition to the traditional VAE loss (reconstruction loss + KL divergence), the Beta-VAE introduces a hyperparameter, β, which scales the KL divergence term. This effectively balances the trade-off between data reconstruction and latent space regularization.

3) Noise Contrastive Priors (NCP): This project employs an energy-based binary classifier trained to distinguish between true posterior samples (q(z|x)) and noise samples (p(z)) in the latent space. This classifier thereby shapes a prior in the latent space that is more expressive than the standard Gaussian prior used in VAEs.

4) Energy Function & Langevin Dynamics: After training the binary classifier, this project also includes an illustrative experiment that uses the classifier as an energy function to perform sampling in the latent space using Langevin Dynamics.

## Hyperparameters
The following hyperparameters can be tuned for the model:

- LATENT_SIZE: The dimensionality of the latent space.
- LR: The learning rate for the optimizer.
- EPOCHS: Number of training epochs for the VAE.
- BETA: The scaling factor for the KL divergence term in the VAE loss.

## Training the Model
Training the VAE model involves two main loss components:

- Reconstruction Loss: Measures how well the reconstructed data matches the original data.
- Kullback-Leibler (KL) Divergence: Regularizes the learned representations by comparing the posterior distribution of the encoder to a prior distribution (usually a standard Gaussian). It is scaled by a hyperparameter beta.
Training the binary classifier involves using a Binary Cross-Entropy (BCE) Loss.

## Results
At the end of the training, the model will produce:

1) The trained parameters of the VAE and the binary classifier.
2) Plots depicting the average and lowest loss during the training of the binary classifier.
   
## Langevin Dynamics Sampling
After training, the project demonstrates how to use the trained binary classifier as an energy function to sample new points in the latent space using Langevin Dynamics. These samples can then be passed through the decoder of the VAE to generate new data samples.
