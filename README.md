# Single-photon-detection neural networks (SPDNNs)

This repository contains the code used for training and analyzing the *Single-Photon-Detection Neural Networks (SPDNNs)*. These networks have been detailed in our study "[Quantum-noise-limited optical neural networks operating at a few quanta per activation](https://arxiv.org/abs/2307.15712)".

In essence, SPDNNs are uniquely designed to detect only a few photons with each dot product or neuron activation during inference. The models trained here can be integrated on various Optical Neural Network (ONN) platforms equipped with single-photon detectors (SPDs). In order to include the stochastic single-photon detection process during training, we employ "*physics-aware stochastic training*" to incorporate the actual stochastic physical process forward pass, while eliminate the stochasicity in the backward pass. This is inspired by the training methods for binary stochastic neurons (see [Bengio et al. (2013)](https://arxiv.org/abs/1308.3432)).

## Repository Contents

This repository includes several directories, each containing specific resources:

### [src](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

The 'src' directory contains Python scripts that define the photon-detection activation function and the SPDNN models.

### [training](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

In the 'training' directory, you will find Jupyter notebooks that provide examples of training SPDNN models.

### [models](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

Trained models from the notebooks are stored in the 'models' directory.

### [test](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

In the 'test' directory, you will find Jupyter notebooks to test the performance of the trained SPDNN models.

### [results](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

Processed test accuracies and intermediate data are saved in the 'results' directory.

### [data](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/data)

The 'data' directory houses external data used for training.

### [plots](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

All generated plots are stored in the 'plots' directory.

### [misc](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/src)

Other utility functions and files can be found in the 'misc' directory.

## Setup Requirements

Refer to the [requirements.txt](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/requirements.txt) file for necessary Python packages. Use the following commands to set up the environment with conda:
```
conda create --name spdnn python=3.9 pip
conda activate spdnn
pip3 install -r requirements.txt
```

## Results

784-$`N_1`$-$`N_2`$-10 denotes the multilayer perceptron (MLP) structure of the SPDNN models with $N$s to be the number of neurons in the hidden layers. **C16** denotes a convolution layer of 16 output channels. For details, see the [paper and supplementary material](https://arxiv.org/abs/2307.15712).

$K$ denotes the number of SPD binary readouts used to compute each SPD activation value.

* MNIST test accuracies for SPDNNs using **coherent** optical matrix-vector multiplers where the information is encoded in optical amplitude, and directly detected by SPDs.

|    Model        |   $K=1$         |   $K=2$         |   $K=3$         |   $K=5$         |   $K=7$         |   $K=10$        |   $K\rightarrow\infty$   |
|:-----------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:------------------------:|
| [784-10-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B10%5D.pth)     | 80.13±0.32%  | 85.65±0.23%  | 87.52±0.20%  | 88.86±0.18%  | 89.40±0.15%  | 89.87±0.15%  | 90.70±0.00%            |
| [784-20-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B20%5D.pth)     | 89.32±0.21%  | 92.46±0.17%  | 93.35±0.17%  | 94.09±0.15%  | 94.40±0.12%  | 94.62±0.11%  | 95.05±0.00%            |
| [784-25-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B25%5D.pth)     | 91.50±0.22%  | 94.14±0.15%  | 94.89±0.14%  | 95.48±0.10%  | 95.69±0.10%  | 95.86±0.10%  | 96.10±0.00%            |
| [784-30-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B30%5D.pth)     | 92.79±0.20%  | 94.92±0.13%  | 95.56±0.12%  | 96.06±0.11%  | 96.26±0.09%  | 96.40±0.10%  | 96.74±0.00%            |
| [784-50-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B50%5D.pth)     | 95.51±0.15%  | 96.89±0.11%  | 97.25±0.09%  | 97.55±0.08%  | 97.66±0.08%  | 97.75±0.07%  | 97.93±0.00%            |
| [784-100-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B100%5D.pth)   | 97.40±0.12%  | 98.15±0.10%  | 98.36±0.07%  | 98.52±0.07%  | 98.58±0.06%  | 98.62±0.06%  | 98.70±0.00%            |
| [784-200-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B200%5D.pth)   | 98.32±0.10%  | 98.77±0.07%  | 98.89±0.06%  | 98.97±0.06%  | 99.01±0.05%  | 99.03±0.05%  | 99.12±0.00%            |
| [784-400-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B400%5D.pth)   | 98.64±0.08%  | 98.95±0.06%  | 99.03±0.05%  | 99.10±0.05%  | 99.12±0.05%  | 99.13±0.04%  | 99.19±0.00%            |
| [784-400-400-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_coh_N%5B400,400%5D.pth) | 98.94±0.07%  | 99.22±0.06%  | 99.29±0.05%  | 99.33±0.04%  | 99.36±0.04%  | 99.37±0.04%  | 99.40±0.00%            |
| [784-C16-400-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_conv16.pth)  | 99.33±0.06%  | 99.45±0.04%  | 99.47±0.04%  | 99.50±0.04%  | 99.51±0.03%  | 99.52±0.03%  | 99.54±0.00%            |

* MNIST test accuracies for SPDNNs using **incoherent** optical matrix-vector multiplers where the information is encoded in light intensity (non-negative).

|    Model        |   $K=1$         |   $K=2$         |   $K=3$         |   $K=5$         |   $K=7$         |   $K=10$        |   $K\rightarrow\infty$   |
|:-----------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:------------------------:|
| [784-10-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N10.pth)       | 78.04±0.29%  | 83.23±0.23%  | 84.78±0.22%  | 86.12±0.16%  | 86.64±0.17%  | 87.09±0.14%  | 87.91±0.00%            |
| [784-20-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N20.pth)       | 86.73±0.23%  | 89.97±0.19%  | 90.97±0.14%  | 91.69±0.14%  | 92.01±0.12%  | 92.21±0.13%  | 92.66±0.00%            |
| [784-50-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N50.pth)       | 93.01±0.16%  | 94.47±0.13%  | 94.93±0.13%  | 95.23±0.09%  | 95.38±0.09%  | 95.48±0.07%  | 95.73±0.00%            |
| [784-100-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N100.pth)     | 95.18±0.15%  | 96.22±0.10%  | 96.53±0.09%  | 96.77±0.09%  | 96.84±0.08%  | 96.90±0.08%  | 97.02±0.00%            |
| [784-200-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N200.pth)     | 96.61±0.13%  | 97.33±0.09%  | 97.54±0.09%  | 97.70±0.08%  | 97.76±0.07%  | 97.80±0.05%  | 97.98±0.00%            |
| [784-300-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N300.pth)     | 96.99±0.13%  | 97.62±0.10%  | 97.77±0.09%  | 97.92±0.06%  | 97.97±0.06%  | 98.01±0.07%  | 98.12±0.00%            |
| [784-400-10](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/main/models/MNIST28x28_incoh_N400.pth)     | 97.30±0.09%  | 97.85±0.09%  | 98.01±0.08%  | 98.15±0.06%  | 98.21±0.06%  | 98.26±0.05%  | 98.41±0.00%            |


## Acknowledgements
Our work owes a lot to earlier studies on binary stochastic neurons, the construction of this repository was inspired by other works, including but not limited to:
* A [GitHub repository](https://github.com/Wizaron/binary-stochastic-neurons/blob/master) that constructs binary stochastic neurons using PyTorch.
* A [comprehensive article](http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html) introducing binary stochastic neurons and showcasing preliminary results.

## How to Cite

If you find our data or code useful for your research, please consider citing the following paper:

> S.-Y. Ma, T. Wang, J. Laydevant, L. G. Wright and P. L. McMahon. "Quantum-noise-limited optical neural networks operating at a few quanta per activation." [arXiv:2307.15712](https://arxiv.org/abs/2307.15712) (2023)

## License

The code in this repository is released under the following license: 

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/mcmahon-lab/Single-Photon-Detection-Neural-Networks/blob/master/license.txt).