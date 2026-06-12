#!/bin/bash

set -ex

# This script tests features intended to be stabilized in 2026. We want to
# ensure they don't regress until then.

# 1. For the new trait solver, we want to:
# - ensure it can build the standard library
#
# FIXME: we also need to ensure it actually bootstraps.

RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ../x build library --stage 1

# 2. For the polonius alpha, we run the UI tests under the polonius
# compare-mode.
#
# Note that we keep the same rustflags to avoid needing to rebuild any stage 1
# artifacts from the previous command. It also tests both features at the same
# time.

RUSTFLAGS_NOT_BOOTSTRAP="-Znext-solver=globally" ../x test tests/ui \
    --compare-mode polonius --stage 1
