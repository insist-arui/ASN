Our codebase is built upon SeFAR. We sincerely thank the authors for their valuable work.
# ASN
# Installation

First, create a new conda environment and activate it:

```bash
conda create -n your_env python=3.7
conda activate your_env
```

Then, install the runtime environment by running:

```bash
bash env.sh
```

# Pretrained Model

Download the pretrained model and place it in the desired directory. Then specify the pretrained model path in the configuration file before training. The checkpoint can be downloaded from: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

---

# Training

Start training by running:

```bash
bash train.sh
```

To train on different datasets or modify the training settings, edit:

```
train.sh
configs/*.yaml
```

If you want to change the parameters of ASN, please edit the `timesformer/config/defaults.py` file.

