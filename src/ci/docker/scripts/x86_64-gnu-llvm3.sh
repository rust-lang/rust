#!/bin/bash

set -ex

##### Test stage 1 #####

../x.py --stage 1 test --skip src/tools/tidy

# Run the `mir-opt` tests again but this time for a 32-bit target.
# This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
# both 32-bit and 64-bit outputs updated by the PR author, before
# the PR is approved and tested for merging.
# It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
# despite having different output on 32-bit vs 64-bit targets.
../x.py --stage 1 test tests/mir-opt --host='' --target=i686-unknown-linux-gnu

# Run `ui-fulldeps` in `--stage=1`, which actually uses the stage0
# compiler, and is sensitive to the addition of new flags.
../x.py --stage 1 test tests/ui-fulldeps

# Rebuild the stdlib with the size optimizations enabled and run tests again.
RUSTFLAGS_NOT_BOOTSTRAP="--cfg feature=\"optimize_for_size\"" ../x.py --stage 1 test \
    library/std library/alloc library/core
