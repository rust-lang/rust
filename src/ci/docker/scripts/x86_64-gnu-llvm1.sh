#!/bin/bash

set -ex

/tmp/add_dummy_commit.sh

../x.py --stage 2 test \
  --skip tests \
  --skip coverage-map \
  --skip coverage-run \
  --skip library \
  --skip tidyselftest
