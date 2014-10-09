// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static A1: uint = 1;
static mut A2: uint = 1;
const A3: uint = 1;

fn main() {
    match 1u {
        A1 => {} //~ ERROR: static variables cannot be referenced in a pattern
        A2 => {} //~ ERROR: static variables cannot be referenced in a pattern
        A3 => {}
        _ => {}
    }
}
