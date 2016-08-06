// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the call operator autoderefs when calling to an object type.

#![allow(unknown_features)]

use std::ops::FnMut;

fn make_adder(x: isize) -> Box<FnMut(isize)->isize + 'static> {
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    Box::new(move |y| { x + y })
}

pub fn main() {
    let mut adder = make_adder(3);
    let z = adder(2);
    println!("{}", z);
    assert_eq!(z, 5);
}
