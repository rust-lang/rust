#!/usr/bin/env python3

"""
This script serves for generating a matrix of jobs that should
be executed on CI.

It reads job definitions from `src/ci/github-actions/jobs.yml`
and filters them based on the event that happened on CI.

Currently, it only supports PR builds.
"""

import json
from pathlib import Path

import yaml

JOBS_YAML_PATH = Path(__file__).absolute().parent.parent / "github-actions" / "jobs.yml"


if __name__ == "__main__":
    with open(JOBS_YAML_PATH) as f:
        jobs = yaml.safe_load(f)
    job_output = jobs["pr"]
    print(f"jobs={json.dumps(job_output)}")
