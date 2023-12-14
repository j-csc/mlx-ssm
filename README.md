# MLX implementation of SSMs

> Repository for learning SSMs using MLX
Planning to implement the following SSMs:
 - [ ] Mamba
 - [ ] S4
 - [ ] S4ND
 Mamba Based on https://arxiv.org/abs/2312.00752

## Mamba Setup
    
```bash
# Download model weights
mkdir pretrained && cd pretrained
huggingface-cli download state-spaces/mamba-130m pytorch_model.bin config.json

cd ..

# Convert to npz format
python convert.py
```

## Resources
- https://ocw.mit.edu/courses/16-30-feedback-control-systems-fall-2010/1bfc976fcead1982d90c5057511e5ef7_MIT16_30F10_lec05.pdf
- https://arxiv.org/pdf/2312.00752.pdf
- https://github.com/state-spaces/s4# mlx-ssm
