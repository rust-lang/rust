# Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -x
rm ~/.cargo/bin/cargo-clippy
cargo install --force --path .

echo "Running integration test for crate ${INTEGRATION}"

git clone --depth=1 https://github.com/${INTEGRATION}.git checkout
cd checkout

function check() {
# run clippy on a project, try to be verbose and trigger as many warnings as possible for greater coverage
  RUST_BACKTRACE=full cargo clippy --all-targets --all-features -- --cap-lints warn -W clippy::pedantic -W clippy::nursery  &> clippy_output
  cat clippy_output
  ! cat clippy_output | grep -q "internal compiler error\|query stack during panic\|E0463"
  if [[ $? != 0 ]]; then
    return 1
  fi
}

case ${INTEGRATION} in
  *)
    check
    ;;
esac
