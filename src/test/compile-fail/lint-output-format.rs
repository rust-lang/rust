// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -F unused_features
// aux-build:lint_output_format.rs

#![allow(deprecated)]

extern crate lint_output_format; //~ ERROR use of unstable library feature
use lint_output_format::{foo, bar}; //~ ERROR use of unstable library feature

fn main() {
    let _x = foo();
    let _y = bar(); //~ ERROR use of unstable library feature
}
