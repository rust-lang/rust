// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the call operator autoderefs when calling a bounded type parameter.

use std::ops::FnMut;

fn call_with_2(x: &fn(isize) -> isize) -> isize
{
    x(2) // look ma, no `*`
}

fn subtract_22(x: isize) -> isize { x - 22 }

pub fn main() {
    let subtract_22: fn(isize) -> isize = subtract_22;
    let z = call_with_2(&subtract_22);
    assert_eq!(z, -20);
}
