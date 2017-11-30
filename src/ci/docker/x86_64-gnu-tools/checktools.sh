#!/bin/sh

# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -eu

X_PY="$1"
TOOLSTATE_FILE="$2"

touch "$TOOLSTATE_FILE"

set +e
python2.7 "$X_PY" test --no-fail-fast \
    src/tools/rls \
    src/tools/rustfmt \
    src/tools/miri \
    src/tools/clippy
TEST_RESULT=$?
set -e

# FIXME: Upload this file to the repository.
cat "$TOOLSTATE_FILE"

# FIXME: After we can properly inform dev-tool maintainers about failure,
#        comment out the `exit 0` below.
if [ "$RUST_RELEASE_CHANNEL" = nightly ]; then
    # exit 0
    true
fi

exit $TEST_RESULT
