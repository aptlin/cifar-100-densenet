# Image classification demo of DenseNet in PyTorch on CIFAR 100

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdll/cifar-100-densenet/demo.ipynb)

## Architecture

### Model Architecture
![Model Architecture](docs/assets/densenet-arch.png)

### Dense Block
<img src="./docs/assets/dense-block.png" width="50%" alt ="Dense block">

## Results

- [Comet.ML Logs](https://www.comet.ml/fastrino/fastrino/11700425158b4ee2a3c02a2fa3e335a1)
- Train F1
  ![Train F1](docs/assets/train-f1.png)
- Train accuracy
  ![Train accuracy](docs/assets/train-accuracy.png)
- Loss
  ![Loss](docs/assets/loss.png)
- Test accuracy & F1
  ![Test Results](docs/assets/test-results.png)


## References

1. Original Paper: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
2. Tutorial: [Implementing DenseNet from Scratch](https://d2l.ai/chapter_convolutional-modern/densenet.html)
3. PyTorch Vision: [Reference Implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
