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

use std::ops::RangeToInclusive;

fn return_range_to() -> RangeToInclusive<i32> {
    return ...1; //~ERROR unexpected token: `...`
                 //~^HELP  use `..` for an exclusive range
                 //~^^HELP or `..=` for an inclusive range
}

pub fn main() {
    let x = ...0;    //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range

    let x = 5...5;   //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range

    for _ in 0...1 {} //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range
}
