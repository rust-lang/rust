# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

curl -so /usr/local/bin/sccache \
  https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
