#!/usr/bin/env python3

"""
This script serves for generating a matrix of jobs that should
be executed on CI.

It reads job definitions from `src/ci/github-actions/jobs.yml`
and filters them based on the event that happened on CI.
"""
import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

CI_DIR = Path(__file__).absolute().parent.parent
JOBS_YAML_PATH = Path(__file__).absolute().parent / "jobs.yml"

Job = Dict[str, Any]


def name_jobs(jobs: List[Dict], prefix: str) -> List[Job]:
    """
    Add a `name` attribute to each job, based on its image and the given `prefix`.
    """
    for job in jobs:
        job["name"] = f"{prefix} - {job['image']}"
    return jobs


def add_base_env(jobs: List[Job], environment: Dict[str, str]) -> List[Job]:
    """
    Prepends `environment` to the `env` attribute of each job.
    The `env` of each job has higher precedence than `environment`.
    """
    for job in jobs:
        env = environment.copy()
        env.update(job.get("env", {}))
        job["env"] = env
    return jobs


class JobType(enum.Enum):
    PR = enum.auto()
    Try = enum.auto()
    Auto = enum.auto()


@dataclasses.dataclass
class GitHubCtx:
    event_name: str
    ref: str
    repository: str


def find_job_type(ctx: GitHubCtx) -> Optional[JobType]:
    if ctx.event_name == "pull_request":
        return JobType.PR
    elif ctx.event_name == "push":
        old_bors_try_build = (
            ctx.ref in ("refs/heads/try", "refs/heads/try-perf") and
            ctx.repository == "rust-lang-ci/rust"
        )
        new_bors_try_build = (
            ctx.ref == "refs/heads/automation/bors/try" and
            ctx.repository == "rust-lang/rust"
        )
        try_build = old_bors_try_build or new_bors_try_build

        if try_build:
            return JobType.Try

        if ctx.ref == "refs/heads/auto" and ctx.repository == "rust-lang-ci/rust":
            return JobType.Auto

    return None


def calculate_jobs(job_type: JobType, job_data: Dict[str, Any]) -> List[Job]:
    if job_type == JobType.PR:
        return add_base_env(name_jobs(job_data["pr"], "PR"), job_data["envs"]["pr"])
    elif job_type == JobType.Try:
        return add_base_env(name_jobs(job_data["try"], "try"), job_data["envs"]["try"])
    elif job_type == JobType.Auto:
        return add_base_env(name_jobs(job_data["auto"], "auto"), job_data["envs"]["auto"])

    return []


def skip_jobs(jobs: List[Dict[str, Any]], channel: str) -> List[Job]:
    """
    Skip CI jobs that are not supposed to be executed on the given `channel`.
    """
    return [j for j in jobs if j.get("only_on_channel", channel) == channel]


def get_github_ctx() -> GitHubCtx:
    return GitHubCtx(
        event_name=os.environ["GITHUB_EVENT_NAME"],
        ref=os.environ["GITHUB_REF"],
        repository=os.environ["GITHUB_REPOSITORY"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open(JOBS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    github_ctx = get_github_ctx()

    job_type = find_job_type(github_ctx)
    logging.info(f"Job type: {job_type}")

    with open(CI_DIR / "channel") as f:
        channel = f.read().strip()

    jobs = []
    if job_type is not None:
        jobs = calculate_jobs(job_type, data)
    jobs = skip_jobs(jobs, channel)

    logging.info(f"Output:\n{yaml.dump(jobs, indent=4)}")
    print(f"jobs={json.dumps(jobs)}")
