#!/bin/bash

set -ex

# Run a subset of tests. Used to run tests in parallel in multiple jobs.

# When this job partition is run as part of PR CI, skip tidy to allow revealing more failures. The
# dedicated `tidy` job failing won't block other PR CI jobs from completing, and so tidy failures
# shouldn't inhibit revealing other failures in PR CI jobs.
if [ "$PR_CI_JOB" == "1" ]; then
  echo "PR_CI_JOB set; skipping tidy"
  SKIP_TIDY="--skip tidy"
fi

# Skip intrinsic-test on LLVM 21 to avoid CI failures.
if [ "$LLVM_VERSION" = "21" ]; then
  echo "LLVM_VERSION is 21; skipping intrinsic-test"
  SKIP_INTRINSICS="--skip library/stdarch/crates/intrinsic-test"
fi

../x.py --stage 2 test \
  ${SKIP_TIDY:+$SKIP_TIDY} \
  ${SKIP_INTRINSICS:+$SKIP_INTRINSICS} \
  --skip compiler \
  --skip src
