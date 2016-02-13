// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:TEST_EXEC_ENV=22
// ignore-emscripten FIXME: issue #31622

use std::env;

pub fn main() {
    assert_eq!(env::var("TEST_EXEC_ENV"), Ok("22".to_string()));
}
