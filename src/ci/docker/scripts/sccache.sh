# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# ignore-tidy-linelength

set -ex

curl -fo /usr/local/bin/sccache \
  https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/2018-04-02-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
