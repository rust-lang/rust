// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

// Make sure that inclusive ranges with `...` syntax don't parse.

#![feature(inclusive_range_syntax, inclusive_range)]

use std::ops::RangeToInclusive;

fn return_range_to() -> RangeToInclusive<i32> {
    return ...1; //~ERROR `...` syntax cannot be used in expressions
                 //~^HELP  Use `..` if you need an exclusive range (a < b)
                 //~^^HELP or `..=` if you need an inclusive range (a <= b)
}

pub fn main() {
    let x = ...0;    //~ERROR `...` syntax cannot be used in expressions
                     //~^HELP  Use `..` if you need an exclusive range (a < b)
                     //~^^HELP or `..=` if you need an inclusive range (a <= b)

    let x = 5...5;   //~ERROR `...` syntax cannot be used in expressions
                     //~^HELP  Use `..` if you need an exclusive range (a < b)
                     //~^^HELP or `..=` if you need an inclusive range (a <= b)

    for _ in 0...1 {} //~ERROR `...` syntax cannot be used in expressions
                     //~^HELP  Use `..` if you need an exclusive range (a < b)
                     //~^^HELP or `..=` if you need an inclusive range (a <= b)
}

