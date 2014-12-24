// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that move restrictions are enforced on overloaded unary operations

fn move_then_borrow<T: Not<T> + Clone>(x: T) {
    !x;

    x.clone();  //~ ERROR: use of moved value
}

fn move_borrowed<T: Not<T>>(x: T, mut y: T) {
    let m = &x;
    let n = &mut y;

    !x;  //~ ERROR: cannot move out of `x` because it is borrowed

    !y;  //~ ERROR: cannot move out of `y` because it is borrowed
}

fn illegal_dereference<T: Not<T>>(mut x: T, y: T) {
    let m = &mut x;
    let n = &y;

    !*m;  //~ ERROR: cannot move out of dereference of `&mut`-pointer

    !*n;  //~ ERROR: cannot move out of dereference of `&`-pointer
}

fn main() {}
