// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// Matching against NaN should result in a warning

use std::f64::NAN;

fn main() {
    let x = NAN;
    match x {
        NAN => {},
        _ => {},
    };
    //~^^^ WARNING unmatchable NaN in pattern, use the is_nan method in a guard instead
    match [x, 1.0] {
        [NAN, _] => {},
        _ => {},
    };
    //~^^^ WARNING unmatchable NaN in pattern, use the is_nan method in a guard instead
}

// At least one error is needed so that compilation fails
#[static_assert]
static b: bool = false; //~ ERROR static assertion failed
