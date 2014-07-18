// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

use std::ops::FnMut;

fn call_it<F:FnMut<(int,int),int>>(y: int, mut f: F) -> int {
    f.call_mut((2, y))
}

pub fn main() {
    let f = |&mut: x: int, y: int| -> int { x + y };
    let z = call_it(3, f);
    println!("{}", z);
    assert_eq!(z, 5);
}

