#!/bin/bash

set -ex

if [ "$READ_ONLY_SRC" = "0" ]; then
    # `core::builder::tests::ci_rustc_if_unchanged_logic` bootstrap test ensures that
    # "download-rustc=if-unchanged" logic don't use CI rustc while there are changes on
    # compiler and/or library. Here we are adding a dummy commit on compiler and running
    # that test to make sure we never download CI rustc with a change on the compiler tree.
    echo "" >> ../compiler/rustc/src/main.rs
    git config --global user.email "dummy@dummy.com"
    git config --global user.name "dummy"
    git add ../compiler/rustc/src/main.rs
    git commit -m "test commit for rust.download-rustc=if-unchanged logic"
    DISABLE_CI_RUSTC_IF_INCOMPATIBLE=0 ../x.py test bootstrap \
        -- core::builder::tests::ci_rustc_if_unchanged_logic
    # Revert the dummy commit
    git reset --hard HEAD~1
fi

# Only run the stage 1 tests on merges, not on PR CI jobs.
if [[ -z "${PR_CI_JOB}" ]]; then
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
fi

# NOTE: intentionally uses all of `x.py`, `x`, and `x.ps1` to make sure they all work on Linux.
../x.py --stage 2 test --skip src/tools/tidy

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
