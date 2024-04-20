#!/usr/bin/env python3

"""
This script serves for generating a matrix of jobs that should
be executed on CI.

It reads job definitions from `src/ci/github-actions/jobs.yml`
and filters them based on the event that happened on CI.

Currently, it only supports PR and try builds.
"""
import enum
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

JOBS_YAML_PATH = Path(__file__).absolute().parent / "jobs.yml"


def name_jobs(jobs: List[Dict], prefix: str) -> List[Dict]:
    for job in jobs:
        job["name"] = f"{prefix} - {job['image']}"
    return jobs


class JobType(enum.Enum):
    PR = enum.auto()
    Try = enum.auto()


def find_job_type(github_ctx: Dict[str, Any]) -> Optional[JobType]:
    event_name = github_ctx["event_name"]
    ref = github_ctx["ref"]
    repository = github_ctx["repository"]

    if event_name == "pull_request":
        return JobType.PR
    elif event_name == "push":
        old_bors_try_build = (
            ref in ("refs/heads/try", "refs/heads/try-perf") and
            repository == "rust-lang-ci/rust"
        )
        new_bors_try_build = (
            ref == "refs/heads/automation/bors/try" and
            repository == "rust-lang/rust"
        )
        try_build = old_bors_try_build or new_bors_try_build

        if try_build:
            return JobType.Try

    return None


def calculate_jobs(job_type: JobType, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if job_type == JobType.PR:
        return name_jobs(job_data["pr"], "PR")
    elif job_type == JobType.Try:
        return name_jobs(job_data["try"], "try")

    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    github_ctx = json.loads(os.environ["GITHUB_CTX"])

    with open(JOBS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    job_type = find_job_type(github_ctx)
    logging.info(f"Job type: {job_type}")

    jobs = []
    if job_type is not None:
        jobs = calculate_jobs(job_type, data)

    logging.info(f"Output:\n{yaml.dump(jobs, indent=4)}")
    print(f"jobs={json.dumps(jobs)}")
