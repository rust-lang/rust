// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// must-compile-successfully

#![feature(exclusive_range_pattern)]
#![warn(unreachable_patterns)]

fn main() {
    // These cases should generate no warning.
    match 10 {
        1..10 => {},
        10 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        9...10 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        10...10 => {},
        _ => {},
    }

    // These cases should generate an "unreachable pattern" warning.
    match 10 {
        1..10 => {},
        9 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        8...9 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        9...9 => {},
        _ => {},
    }
}
