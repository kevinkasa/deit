# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
import json
from pathlib import Path

import main as classification
import submitit

import wandb
import optuna


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for DeiT", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--ncpus", default = 10, type = int, help = "Number of cpus to request per each gpu")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    # parser.add_argument('--mem', type=int)
    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    parser.add_argument('--param_tune', default=False, type=bool, help='Is this run tuning hyperparameters')
    parser.add_argument('--ntrials', default=10, type=int, help='Number of Optuna trials to run')
    parser.add_argument('--tuning_params', type=json.loads)
    
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main, hparam_tune

        self._setup_gpu_args()
        # run hyperparameter tuning or regular training
        if self.args.param_tune:
            hparam_tune.hparam_search(self.args)
        else:
            main.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        print(self.args.gpu)
        # set up W&B, if not tuning
        if self.args.project and not self.args.param_tune:
            if self.args.rank == 0:
                if Path(self.args.wandb_file).exists():
                    resume_id = Path(self.args.wandb_file).read_text()
                    wandb.init(project=self.args.project, name=self.args.exp_name, notes=self.args.notes,
                               tags=self.args.tags, id=resume_id, resume='allow', group='group_' + self.args.exp_name,
                               dir=self.args.output_dir)
                    wandb.config.update(self.args, allow_val_change=True)
                else:
                    wandb.init(project=self.args.project, name=self.args.exp_name, notes=self.args.notes,
                               tags=self.args.tags, group='group_' + self.args.exp_name, dir=self.args.output_dir)
                    wandb.config.update(self.args)
                    Path(self.args.wandb_file).write_text(str(wandb.run.id))


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
	    exclude='gpu179',
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.ncpus,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_srun_args={'--mem ' + str(40 * num_gpus_per_node) + 'GB'},
        **kwargs
    )

    executor.update_parameters(name="deit")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
