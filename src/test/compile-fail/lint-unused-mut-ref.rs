// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![deny(unused_mut)]

fn foo(any: &mut i32) {} //~ ERROR: variable does not need to be mutable

fn bar(_any: &mut i32) {}

fn main() {
    let mut x = 42;
    foo(&mut x);

    let mut something = Some(43);

    match something {
        Some(ref mut i) => {}, //~ ERROR: variable does not need to be mutable
        None => {},
    }

    // Positive cases
    bar(&mut x);

    match something {
        Some(ref mut _i) => {},
        None => {},
    }
}
