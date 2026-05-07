#!/bin/bash

set -ex

# NOTE: intentionally uses `x`, and `x.ps1` to make sure they work on Linux.
#       Make sure that `x.py` is tested elsewhere.

# Run the `mir-opt` tests again but this time for a 32-bit target.
# This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
# both 32-bit and 64-bit outputs updated by the PR author, before
# the PR is approved and tested for merging.
# It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
# despite having different output on 32-bit vs 64-bit targets.
../x --stage 2 test tests/mir-opt --host='' --target=i686-unknown-linux-gnu

# Run the UI test suite in `--pass=check` mode, to ensure it continues to work.
../x.ps1 --stage 2 test tests/ui --pass=check --host='' --target=i686-unknown-linux-gnu

# Rebuild the stdlib using the new trait solver, to ensure it doesn't regress
# until stabilization.
RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ../x --stage 1 build library \
    --host='' --target=i686-unknown-linux-gnu
