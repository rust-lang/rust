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
cd clippy_lints && cargo test && cd ..
cd rustc_tools_util && cargo test && cd ..
cd clippy_dev && cargo test && cd ..

# Perform various checks for lint registration
./util/dev update_lints --check
cargo +nightly fmt --all -- --check

# Add bin to PATH for windows
PATH=$PATH:$(rustc --print sysroot)/bin

CLIPPY="`pwd`/target/debug/cargo-clippy clippy"
# run clippy on its own codebase...
${CLIPPY} --all-targets --all-features -- -D clippy::all -D clippy::internal -Dclippy::pedantic
# ... and some test directories
for dir in clippy_workspace_tests clippy_workspace_tests/src clippy_workspace_tests/subcrate clippy_workspace_tests/subcrate/src clippy_dev rustc_tools_util
do
    cd ${dir}
    ${CLIPPY} -- -D clippy::all -D clippy::pedantic
    cd -
done


# test --manifest-path
${CLIPPY} --manifest-path=clippy_workspace_tests/Cargo.toml -- -D clippy::all
cd clippy_workspace_tests/subcrate && ${CLIPPY} --manifest-path=../Cargo.toml -- -D clippy::all && cd ../..
set +x
