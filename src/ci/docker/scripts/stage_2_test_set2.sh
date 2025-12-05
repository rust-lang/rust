#!/bin/bash

set -ex

# Run a subset of tests. Used to run tests in parallel in multiple jobs.

../x.py --stage 2 test \
  --skip tests \
  --skip coverage-map \
  --skip coverage-run \
  --skip library \
  --skip tidyselftest
