#!/usr/bin/env python3

"""
This script serves for generating a matrix of jobs that should
be executed on CI.

It reads job definitions from `src/ci/github-actions/jobs.yml`
and filters them based on the event that happened on CI.
"""
import dataclasses
import json
import logging
import os
import re
import typing
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
            "refs/heads/automation/bors/try"
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
        return add_base_env(name_jobs(job_data["pr"], "PR"), job_data["envs"]["pr"])
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
                job = [j for j in job_data["auto"] if j["image"] == custom_job]
                if not job:
                    unknown_jobs.append(custom_job)
                    continue
                jobs.append(job[0])
            if unknown_jobs:
                raise Exception(f"Custom job(s) `{unknown_jobs}` not found in auto jobs")

        return add_base_env(name_jobs(jobs, "try"), job_data["envs"]["try"])
    elif isinstance(run_type, AutoRunType):
        return add_base_env(name_jobs(job_data["auto"], "auto"), job_data["envs"]["auto"])

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
        commit_message=commit_message
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


# Add new function before main:
def substitute_github_vars(jobs: list) -> list:
    """Replace GitHub context variables with environment variables in job configs."""
    for job in jobs:
        if "os" in job:
            job["os"] = job["os"].replace(
                "${{ github.run_id }}",
                os.environ["GITHUB_RUN_ID"]
            ).replace(
                "${{ github.run_attempt }}",
                os.environ["GITHUB_RUN_ATTEMPT"]
            )
    return jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open(JOBS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    github_ctx = get_github_ctx()

    run_type = find_run_type(github_ctx)
    logging.info(f"Job type: {run_type}")

    with open(CI_DIR / "channel") as f:
        channel = f.read().strip()

    jobs = []
    if run_type is not None:
        jobs = calculate_jobs(run_type, data)
    jobs = skip_jobs(jobs, channel)
    jobs = substitute_github_vars(jobs)


    if not jobs:
        raise Exception("Scheduled job list is empty, this is an error")

    run_type = format_run_type(run_type)

    logging.info(f"Output:\n{yaml.dump(dict(jobs=jobs, run_type=run_type), indent=4)}")
    print(f"jobs={json.dumps(jobs)}")
    print(f"run_type={run_type}")
