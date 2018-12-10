# Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


set -ex

echo "Running clippy base tests"

PATH=$PATH:./node_modules/.bin
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  remark -f *.md > /dev/null
fi
# build clippy in debug mode and run tests
cargo build --features debugging
cargo test --features debugging
# for faster build, share target dir between subcrates
export CARGO_TARGET_DIR=`pwd`/target/
cd clippy_lints && cargo test && cd ..
cd rustc_tools_util && cargo test && cd ..
cd clippy_dev && cargo test && cd ..

# Perform various checks for lint registration
./util/dev update_lints --check
cargo +nightly fmt --all -- --check
