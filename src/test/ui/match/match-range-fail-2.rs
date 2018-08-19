// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(exclusive_range_pattern)]

fn main() {
    match 5 {
        6 ..= 1 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than or equal to upper

    match 5 {
        0 .. 0 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than upper

    match 5u64 {
        0xFFFF_FFFF_FFFF_FFFF ..= 1 => { }
        _ => { }
    };
    //~^^^ ERROR lower range bound must be less than or equal to upper
}
