// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that extern fn pointers implement the full range of Fn traits.

#![feature(unboxed_closures)]
#![feature(unboxed_closures)]

use std::ops::{Fn,FnMut,FnOnce};

fn square(x: int) -> int { x * x }

fn call_it<F:Fn(int)->int>(f: &F, x: int) -> int {
    f(x)
}

fn call_it_mut<F:FnMut(int)->int>(f: &mut F, x: int) -> int {
    f(x)
}

fn call_it_once<F:FnOnce(int)->int>(f: F, x: int) -> int {
    f(x)
}

fn main() {
    let x = call_it(&square, 22);
    let y = call_it_mut(&mut square, 22);
    let z = call_it_once(square, 22);
    assert_eq!(x, square(22));
    assert_eq!(y, square(22));
    assert_eq!(z, square(22));
}

