#!/usr/bin/env python3

"""
This script contains CI functionality.
It can be used to generate a matrix of jobs that should
be executed on CI, or run a specific CI job locally.

It reads job definitions from `src/ci/github-actions/jobs.yml`.
"""

import argparse
import dataclasses
import json
import logging
import os
import re
import subprocess
import typing
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

CI_DIR = Path(__file__).absolute().parent.parent
JOBS_YAML_PATH = Path(__file__).absolute().parent / "jobs.yml"

Job = Dict[str, Any]


def add_job_properties(jobs: List[Dict], prefix: str) -> List[Job]:
    """
    Modify the `name` attribute of each job, based on its base name and the given `prefix`.
    Add an `image` attribute to each job, based on its image.
    """
    modified_jobs = []
    for job in jobs:
        # Create a copy of the `job` dictionary to avoid modifying `jobs`
        new_job = dict(job)
        new_job["image"] = get_job_image(new_job)
        new_job["name"] = f"{prefix} - {new_job['name']}"
        modified_jobs.append(new_job)
    return modified_jobs


def add_base_env(jobs: List[Job], environment: Dict[str, str]) -> List[Job]:
    """
    Prepends `environment` to the `env` attribute of each job.
    The `env` of each job has higher precedence than `environment`.
    """
    modified_jobs = []
    for job in jobs:
        env = environment.copy()
        env.update(job.get("env", {}))

        new_job = dict(job)
        new_job["env"] = env
        modified_jobs.append(new_job)
    return modified_jobs


@dataclasses.dataclass
class PRRunType:
    pass


@dataclasses.dataclass
class TryRunType:
    custom_jobs: List[str]


@dataclasses.dataclass
class AutoRunType:
    pass


WorkflowRunType = typing.Union[PRRunType, TryRunType, AutoRunType]


@dataclasses.dataclass
class GitHubCtx:
    event_name: str
    ref: str
    repository: str
    commit_message: Optional[str]


def get_custom_jobs(ctx: GitHubCtx) -> List[str]:
    """
    Tries to parse names of specific CI jobs that should be executed in the form of
    try-job: <job-name>
    from the commit message of the passed GitHub context.
    """
    if ctx.commit_message is None:
        return []

    regex = re.compile(r"^try-job: (.*)", re.MULTILINE)
    jobs = []
    for match in regex.finditer(ctx.commit_message):
        jobs.append(match.group(1))
    return jobs


def find_run_type(ctx: GitHubCtx) -> Optional[WorkflowRunType]:
    if ctx.event_name == "pull_request":
        return PRRunType()
    elif ctx.event_name == "push":
        try_build = ctx.ref in (
            "refs/heads/try",
            "refs/heads/try-perf",
            "refs/heads/automation/bors/try",
        )

        # Unrolled branch from a rollup for testing perf
        # This should **not** allow custom try jobs
        is_unrolled_perf_build = ctx.ref == "refs/heads/try-perf"

        if try_build:
            custom_jobs = []
            if not is_unrolled_perf_build:
                custom_jobs = get_custom_jobs(ctx)
            return TryRunType(custom_jobs=custom_jobs)

        if ctx.ref == "refs/heads/auto":
            return AutoRunType()

    return None


def calculate_jobs(run_type: WorkflowRunType, job_data: Dict[str, Any]) -> List[Job]:
    if isinstance(run_type, PRRunType):
        return add_base_env(
            add_job_properties(job_data["pr"], "PR"), job_data["envs"]["pr"]
        )
    elif isinstance(run_type, TryRunType):
        jobs = job_data["try"]
        custom_jobs = run_type.custom_jobs
        if custom_jobs:
            if len(custom_jobs) > 10:
                raise Exception(
                    f"It is only possible to schedule up to 10 custom jobs, "
                    f"received {len(custom_jobs)} jobs"
                )

            jobs = []
            unknown_jobs = []
            for custom_job in custom_jobs:
                job = [j for j in job_data["auto"] if j["name"] == custom_job]
                if not job:
                    unknown_jobs.append(custom_job)
                    continue
                jobs.append(job[0])
            if unknown_jobs:
                raise Exception(
                    f"Custom job(s) `{unknown_jobs}` not found in auto jobs"
                )

        return add_base_env(add_job_properties(jobs, "try"), job_data["envs"]["try"])
    elif isinstance(run_type, AutoRunType):
        return add_base_env(
            add_job_properties(job_data["auto"], "auto"), job_data["envs"]["auto"]
        )

    return []


def skip_jobs(jobs: List[Dict[str, Any]], channel: str) -> List[Job]:
    """
    Skip CI jobs that are not supposed to be executed on the given `channel`.
    """
    return [j for j in jobs if j.get("only_on_channel", channel) == channel]


def get_github_ctx() -> GitHubCtx:
    event_name = os.environ["GITHUB_EVENT_NAME"]

    commit_message = None
    if event_name == "push":
        commit_message = os.environ["COMMIT_MESSAGE"]
    return GitHubCtx(
        event_name=event_name,
        ref=os.environ["GITHUB_REF"],
        repository=os.environ["GITHUB_REPOSITORY"],
        commit_message=commit_message,
    )


def format_run_type(run_type: WorkflowRunType) -> str:
    if isinstance(run_type, PRRunType):
        return "pr"
    elif isinstance(run_type, AutoRunType):
        return "auto"
    elif isinstance(run_type, TryRunType):
        return "try"
    else:
        raise AssertionError()


def get_job_image(job: Job) -> str:
    """
    By default, the Docker image of a job is based on its name.
    However, it can be overridden by its IMAGE environment variable.
    """
    env = job.get("env", {})
    # Return the IMAGE environment variable if it exists, otherwise return the job name
    return env.get("IMAGE", job["name"])


def is_linux_job(job: Job) -> bool:
    return "ubuntu" in job["os"]


def find_linux_job(job_data: Dict[str, Any], job_name: str, pr_jobs: bool) -> Job:
    candidates = job_data["pr"] if pr_jobs else job_data["auto"]
    jobs = [job for job in candidates if job.get("name") == job_name]
    if len(jobs) == 0:
        available_jobs = "\n".join(
            sorted(job["name"] for job in candidates if is_linux_job(job))
        )
        raise Exception(f"""Job `{job_name}` not found in {'pr' if pr_jobs else 'auto'} jobs.
The following jobs are available:
{available_jobs}""")
    assert len(jobs) == 1

    job = jobs[0]
    if not is_linux_job(job):
        raise Exception("Only Linux jobs can be executed locally")
    return job


def run_workflow_locally(job_data: Dict[str, Any], job_name: str, pr_jobs: bool):
    DOCKER_DIR = Path(__file__).absolute().parent.parent / "docker"

    job = find_linux_job(job_data, job_name=job_name, pr_jobs=pr_jobs)

    custom_env = {}
    # Replicate src/ci/scripts/setup-environment.sh
    # Adds custom environment variables to the job
    if job_name.startswith("dist-"):
        if job_name.endswith("-alt"):
            custom_env["DEPLOY_ALT"] = "1"
        else:
            custom_env["DEPLOY"] = "1"
    custom_env.update({k: str(v) for (k, v) in job.get("env", {}).items()})

    args = [str(DOCKER_DIR / "run.sh"), get_job_image(job)]
    env_formatted = [f"{k}={v}" for (k, v) in sorted(custom_env.items())]
    print(f"Executing `{' '.join(env_formatted)} {' '.join(args)}`")

    env = os.environ.copy()
    env.update(custom_env)

    subprocess.run(args, env=env)


def calculate_job_matrix(job_data: Dict[str, Any]):
    github_ctx = get_github_ctx()

    run_type = find_run_type(github_ctx)
    logging.info(f"Job type: {run_type}")

    with open(CI_DIR / "channel") as f:
        channel = f.read().strip()

    jobs = []
    if run_type is not None:
        jobs = calculate_jobs(run_type, job_data)
    jobs = skip_jobs(jobs, channel)

    if not jobs:
        raise Exception("Scheduled job list is empty, this is an error")

    run_type = format_run_type(run_type)

    logging.info(f"Output:\n{yaml.dump(dict(jobs=jobs, run_type=run_type), indent=4)}")
    print(f"jobs={json.dumps(jobs)}")
    print(f"run_type={run_type}")


def create_cli_parser():
    parser = argparse.ArgumentParser(
        prog="ci.py", description="Generate or run CI workflows"
    )
    subparsers = parser.add_subparsers(
        help="Command to execute", dest="command", required=True
    )
    subparsers.add_parser(
        "calculate-job-matrix",
        help="Generate a matrix of jobs that should be executed in CI",
    )
    run_parser = subparsers.add_parser(
        "run-local", help="Run a CI jobs locally (on Linux)"
    )
    run_parser.add_argument(
        "job_name",
        help="CI job that should be executed. By default, a merge (auto) "
        "job with the given name will be executed",
    )
    run_parser.add_argument(
        "--pr", action="store_true", help="Run a PR job instead of an auto job"
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open(JOBS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    parser = create_cli_parser()
    args = parser.parse_args()

    if args.command == "calculate-job-matrix":
        calculate_job_matrix(data)
    elif args.command == "run-local":
        run_workflow_locally(data, args.job_name, args.pr)
    else:
        raise Exception(f"Unknown command {args.command}")
