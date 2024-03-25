# <span style="font-size:larger;">MACE with automatic differrention based Hessians</span>

### Based on MACE (15.03.2024):
[![GitHub release](https://img.shields.io/github/release/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/releases/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPs2022-blue)](https://openreview.net/forum?id=YPpSngE-ZU)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/issues/)
[![Documentation Status](https://readthedocs.org/projects/mace/badge/)](https://mace-docs.readthedocs.io/en/latest/)

## Table of contents 

- [About MACE](#about-mace)
- [Development - based on the [MACE](https://github.com/ACEsuit/mace) code](#documentation)
- [Installation MACE_HESSIAN](#installation)
- [Usage](#usage)
  - [General Usage](#general-usage)
  - [How to Obtain a Hessian](#how-to-obtain-a-hessian)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Tutorial](#tutorial)
- [Weights and Biases](#weights-and-biases-for-experiment-tracking)
- [Development](#development)
- [Pretrained Foundation Models with Hessian implementation](#pretrained-foundation-models)
  - [MACE-MP: Materials Project Force Fields](#mace-mp-materials-project-force-fields)
  - [MACE-OFF: Transferable Organic Force Fields](#mace-off-transferable-organic-force-fields)
- [References](#references)
- [Contact](#contact)
- [License](#license)

## About MACE

MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing
and can be found [here](https://github.com/ACEsuit/mace).

## Documentation
A partial documentation for MACE is available at: https://mace-docs.readthedocs.io

## Installation

Requirements:

- Python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.12 **(training with float64 is not supported with PyTorch 2.1)**.

### pip installation from source

To install via `pip`, follow the steps below:

```sh
# Create a virtual environment and activate it
python -m venv mace-venv
source mace-venv/bin/activate

# Install PyTorch (for example, for CUDA 11.6 [cu116])
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Clone and install MACE (and all required packages)
git clone https://github.com/Nilsgoe/mace_hessian.git
pip install ./mace_hessian
```



## Usage

### General Usage
The general usage is the same as [MACE](https://github.com/ACEsuit/mace) from the 15.03.2024.

### How to Obtain a Hessian
To obtain the hessian the following commands can be use:
```py
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda', )#device='cpu')
atoms.calc = calc
hessian=calc.get_hessian(atoms=atoms)
print("h:",hessian)
```
A [file](https://github.com/Nilsgoe/mace_hessian/tree/main/tests/test_mace_hessian.py) for a larger test and a comparison with numerical derivatives is also provided.


### Training

To train a MACE model, you can use the `mace_run_train` script, which should be in the usual place that pip places binaries (or you can explicitly run `python3 <path_to_cloned_dir>/mace/cli/run_train.py`)

```sh
mace_run_train \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
```

To give a specific validation set, use the argument `--valid_file`. To set a larger batch size for evaluating the validation set, specify `--valid_batch_size`.

To control the model's size, you need to change `--hidden_irreps`. For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`.

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. To label them in the training set, set `config_type=IsolatedAtom` in their info fields. If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression.

If the keyword `--swa` is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_swa` epochs). This setting usually helps lower the energy errors.

The precision can be changed using the keyword `--default_dtype`, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).

The keywords `--batch_size` and `--max_num_epochs` should be adapted based on the size of the training set. The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules. In this example, to make the code ignore stress on molecules, append to your molecules configuration a `config_stress_weight = 0.0`.
'''
To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify `--device=mps`.

### Evaluation

To evaluate your MACE model on an XYZ file, run the `mace_eval_configs`:

```sh
mace_eval_configs \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```
'''
## Tutorial



## Pretrained Foundation Models with Hessian implementation

### MACE-MP: Materials Project Force Fields

We have collaborated with the Materials Project (MP) to train a universal MACE potential covering 89 elements on 1.6 M bulk crystals in the [MPTrj dataset](https://figshare.com/articles/dataset/23713842) selected from MP relaxation trajectories.
The models are releaed on GitHub at https://github.com/ACEsuit/mace-mp.
If you use them please cite [our paper](https://arxiv.org/abs/2401.00096) which also contains an large range of example applications and benchmarks.

#### Example usage in ASE'''

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### MACE-OFF: Transferable Organic Force Fields - will be coming soon

There is a series (small, medium, large) transferable organic force fields. These can be used for the simulation of organic molecules, crystals and molecular liquids, or as a starting point for fine-tuning on a new dataset. The models are released under the [ASL license](https://github.com/gabor1/ASL). 
The models are releaed on GitHub at https://github.com/ACEsuit/mace-off.
If you use them please cite [our paper](https://arxiv.org/abs/2312.15211) which also contains detailed benchmarks and example applications.

#### Example usage in ASE
```py
from mace.calculators import mace_off
from ase import build

atoms = build.molecule('H2O')
calc = mace_off(model="medium", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

## Development - based on the [MACE](https://github.com/ACEsuit/mace) code

This is still underfull development and therefore the code can and will change drasticlly


## References

If you use this code, please cite our papers:

```text
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at goennheimer@fhi.mpg.de.

For bugs or feature requests, please use [GitHub Issues](https://github.com/Nilsgoe/mace_hessian/issues).

## License

MACE_HESSIAN based on MACE and is published and distributed under the [MIT License](MIT.md).
