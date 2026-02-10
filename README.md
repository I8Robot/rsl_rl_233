# RSL RL (Custom Fork - v2.3.3)

> 基于 [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) v2.3.3 的自定义版本，添加了 **ClearML** 作为日志记录和模型管理平台的支持。

A fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac Gym.

### 主要修改

- ✅ 添加 ClearML 支持（日志记录 & 模型管理）

## 安装

### 方式一：直接从 GitHub 安装

> ⚠️ **重要**：如果环境中已安装 IsaacLab，请务必使用 `--no-deps` 参数，避免依赖冲突。

```bash
pip install --no-deps git+https://github.com/I8Robot/rsl_rl_233.git
```

### 方式二：克隆仓库 + 开发模式安装（推荐，方便继续修改代码）

```bash
git clone https://github.com/I8Robot/rsl_rl_233.git
cd rsl_rl_233
pip install --no-deps -e .
```

### 注意事项

- 本包基于 `rsl-rl-lib==2.3.3`，安装后会替换环境中已有的 `rsl-rl-lib`
- 使用 `--no-deps` 可防止 pip 升级 `torch`、`onnx`、`protobuf` 等依赖导致与 IsaacLab 冲突
- 如果是全新环境（无 IsaacLab），可以不加 `--no-deps`：`pip install git+https://github.com/I8Robot/rsl_rl_233.git`

### 支持的日志框架

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/
* **ClearML**: https://clear.ml/ （本 fork 新增）


## Contribution Guidelines

For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. Please make sure that your code is well-documented and follows the guidelines.

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
- [black](https://black.readthedocs.io/en/stable/): The uncompromising code formatter.
- [flake8](https://flake8.pycqa.org/en/latest/): A wrapper around PyFlakes, pycodestyle, and McCabe complexity checker.

Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:

```bash
# for installation (only once)
pre-commit install
# for running
pre-commit run --all-files
```

## Citing

**We are working on writing a white paper for this library.** Until then, please cite the following work
if you use this library for your research:

```text
@InProceedings{rudin2022learning,
  title = 	 {Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author =       {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle = 	 {Proceedings of the 5th Conference on Robot Learning},
  pages = 	 {91--100},
  year = 	 {2022},
  volume = 	 {164},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v164/rudin22a.html},
}
```

If you use the library with curiosity-driven exploration (random network distillation), please cite:

```text
@InProceedings{schwarke2023curiosity,
  title = 	 {Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks},
  author =       {Schwarke, Clemens and Klemm, Victor and Boon, Matthijs van der and Bjelonic, Marko and Hutter, Marco},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {2594--2610},
  year = 	 {2023},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v229/schwarke23a.html},
}
```

If you use the library with symmetry augmentation, please cite:

```text
@InProceedings{mittal2024symmetry,
  author={Mittal, Mayank and Rudin, Nikita and Klemm, Victor and Allshire, Arthur and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Symmetry Considerations for Learning Task Symmetric Robot Policies},
  year={2024},
  pages={7433-7439},
  doi={10.1109/ICRA57147.2024.10611493}
}
```
