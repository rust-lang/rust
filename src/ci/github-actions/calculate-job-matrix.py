#!/usr/bin/env python3

"""
This script serves for generating a matrix of jobs that should
be executed on CI.

It reads job definitions from `src/ci/github-actions/jobs.yml`
and filters them based on the event that happened on CI.

Currently, it only supports PR and try builds.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import yaml

JOBS_YAML_PATH = Path(__file__).absolute().parent / "jobs.yml"


def name_jobs(jobs: List[Dict], prefix: str) -> List[Dict]:
    for job in jobs:
        job["name"] = f"{prefix} - {job['image']}"
    return jobs


if __name__ == "__main__":
    github_ctx = json.loads(os.environ["GITHUB_CTX"])

    with open(JOBS_YAML_PATH) as f:
        data = yaml.safe_load(f)

    event_name = github_ctx["event_name"]
    ref = github_ctx["ref"]
    repository = github_ctx["repository"]

    old_bors_try_build = (
        ref in ("refs/heads/try", "refs/heads/try-perf") and
        repository == "rust-lang-ci/rust"
    )
    new_bors_try_build = (
        ref == "refs/heads/automation/bors/try" and
        repository == "rust-lang/rust"
    )
    try_build = old_bors_try_build or new_bors_try_build

    jobs = []
    # Pull request CI jobs. Their name is 'PR - <image>'
    if event_name == "pull_request":
        jobs = name_jobs(data["pr"], "PR")
    # Try builds
    elif event_name == "push" and try_build:
        jobs = name_jobs(data["try"], "try")

    print(f"Output:\n{json.dumps(jobs, indent=4)}", file=sys.stderr)
    print(f"jobs={json.dumps(jobs)}")
