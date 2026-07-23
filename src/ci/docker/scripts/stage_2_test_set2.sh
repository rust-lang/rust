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

# The combination of rust.randomize-layout and LLVM 21 is causing random
# SIGSEGV in the `hir-def` rust-analyzer tests. Since it seems to be fixed in
# LLVM 22, let's just disable this for now since it has been disruptive for
# CI. See https://github.com/rust-lang/rust/issues/156460.
if [[ "$CI_JOB_NAME" == *"llvm-21"* ]]; then
  echo "CI_JOB_NAME contains 'llvm-21'; skipping rust-analyzer"
  SKIP_RUST_ANALYZER="--skip src/tools/rust-analyzer"
fi

# Skip intrinsic-test on LLVM 21 to avoid CI failures.
if [ "$LLVM_VERSION" = "21" ]; then
  echo "LLVM_VERSION is 21; skipping intrinsic-test"
  SKIP_INTRINSICS="--skip library/stdarch/crates/intrinsic-test"
fi

../x.py --stage 2 test \
  ${SKIP_TIDY:+$SKIP_TIDY} \
  ${SKIP_INTRINSICS:+$SKIP_INTRINSICS} \
  ${SKIP_RUST_ANALYZER:+$SKIP_RUST_ANALYZER} \
  --skip tests \
  --skip library \
  --skip tidyselftest
