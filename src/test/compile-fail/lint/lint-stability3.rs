// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_stability.rs
// error-pattern: use of deprecated item

#![deny(deprecated)]
#![allow(warnings)]

#[macro_use]
extern crate lint_stability;

use lint_stability::*;

fn main() {
    macro_test_arg_nested!(deprecated_text);
}
