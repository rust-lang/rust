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

use std::ops::{Fn,FnMut,FnOnce};

fn square(x: isize) -> isize { x * x }

fn call_it<F:Fn(isize)->isize>(f: &F, x: isize) -> isize {
    f(x)
}

fn call_it_mut<F:FnMut(isize)->isize>(f: &mut F, x: isize) -> isize {
    f(x)
}

fn call_it_once<F:FnOnce(isize)->isize>(f: F, x: isize) -> isize {
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
