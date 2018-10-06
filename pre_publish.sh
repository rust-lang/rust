#!/bin/bash

# Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


set -e

./util/update_lints.py

# add all changed files
git add .
git commit -m "Bump the version"

set +e

echo "Running \`cargo fmt\`.."

cd clippy_lints && cargo fmt -- --write-mode=overwrite && cd ..
cargo fmt -- --write-mode=overwrite

echo "Running tests to make sure \`cargo fmt\` did not break anything.."

cargo test

echo "If the tests passed, review and commit the formatting changes and remember to add a git tag."
