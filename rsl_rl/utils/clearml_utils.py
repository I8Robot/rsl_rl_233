# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter


try:
    from clearml import Task
except ModuleNotFoundError:
    raise ModuleNotFoundError("ClearML is required. Please install with 'pip install clearml'.")

class ClearMLSummaryWriter(SummaryWriter):
    """Summary writer for ClearML."""
    
    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs=flush_secs)
        
        # Get the run name
        run_name = os.path.split(log_dir)[-1]
        
        try:
            project = cfg["clearml_project"]
        except KeyError:
            raise KeyError("Please specify clearml_project in the runner config, e.g. legged_gym.")
            
        # 注意：不需要wandb使用的 entity 参数，clearml会自动读取 ~/clearml.conf 里的配置      
        self.task = Task.init(project_name=project, 
                         task_name=run_name,
                         output_uri=True,
                         reuse_last_task_id=False  # 确保每次运行创建一个新任务，类似 wandb 的逻辑
                         )

        self.task.connect({"log_dir": log_dir})
        self.task.connect(cfg)
        
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }
        
    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        try:
            env_dict = env_cfg.to_dict()
        except Exception:
            env_dict = asdict(env_cfg)
            
        self.task.connect({
            "runner_cfg": runner_cfg,
            "policy_cfg": policy_cfg,
            "alg_cfg": alg_cfg,
            "env_cfg":env_dict
        })
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(tag, scalar_value, global_step, walltime, new_style)
        
    def stop(self):
        self.task.close()
        
    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)
        
    def save_model(self, model_path, iter):
        self.task.upload_artifact(
            name="latest_checkpoint", 
            artifact_object=model_path
        )
        self.task.upload_artifact(artifact_object=model_path, name=os.path.basename(model_path))
        
        #self.task.update_output_model(model_path=model_path, name=f"model_iter_{iter}", iteration=iter)
        
    def save_file(self, path, iter=None):
        self.task.upload_artifact(artifact_object=path, name=os.path.basename(path))
        
    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
