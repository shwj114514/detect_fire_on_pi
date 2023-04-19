def test_torch():
    import torch
    import torch.nn as nn
    print(f"torch版本为{torch.__version__}")
    ceshi=torch.randn(1,3,224,224)
    print(f"输入tensor维度={ceshi.shape}")
    linear=nn.Linear(224,3)
    output=linear(ceshi)
    print(f"输出tensor维度={output.shape}")

if __name__ == '__main__':
    test_torch()