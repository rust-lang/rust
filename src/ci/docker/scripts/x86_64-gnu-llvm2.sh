#!/bin/bash

set -ex

/tmp/add_dummy_commit.sh

##### Test stage 2 #####

../x.py --stage 2 test \
  --skip compiler \
  --skip src

# Run the `mir-opt` tests again but this time for a 32-bit target.
# This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
# both 32-bit and 64-bit outputs updated by the PR author, before
# the PR is approved and tested for merging.
# It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
# despite having different output on 32-bit vs 64-bit targets.
../x --stage 2 test tests/mir-opt --host='' --target=i686-unknown-linux-gnu

# Run the UI test suite again, but in `--pass=check` mode
#
# This is intended to make sure that both `--pass=check` continues to
# work.
../x.ps1 --stage 2 test tests/ui --pass=check --host='' --target=i686-unknown-linux-gnu
