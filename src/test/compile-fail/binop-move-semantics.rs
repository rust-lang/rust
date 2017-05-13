// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that move restrictions are enforced on overloaded binary operations

use std::ops::Add;

fn double_move<T: Add<Output=()>>(x: T) {
    x
    +
    x;  //~ ERROR: use of moved value
}

fn move_then_borrow<T: Add<Output=()> + Clone>(x: T) {
    x
    +
    x.clone();  //~ ERROR: use of moved value
}

fn move_borrowed<T: Add<Output=()>>(x: T, mut y: T) {
    let m = &x;
    let n = &mut y;

    x  //~ ERROR: cannot move out of `x` because it is borrowed
    +
    y;  //~ ERROR: cannot move out of `y` because it is borrowed
}

fn illegal_dereference<T: Add<Output=()>>(mut x: T, y: T) {
    let m = &mut x;
    let n = &y;

    *m  //~ ERROR: cannot move out of borrowed content
    +
    *n;  //~ ERROR: cannot move out of borrowed content
}

struct Foo;

impl<'a, 'b> Add<&'b Foo> for &'a mut Foo {
    type Output = ();

    fn add(self, _: &Foo) {}
}

impl<'a, 'b> Add<&'b mut Foo> for &'a Foo {
    type Output = ();

    fn add(self, _: &mut Foo) {}
}

fn mut_plus_immut() {
    let mut f = Foo;

    &mut f
    +
    &f;  //~ ERROR: cannot borrow `f` as immutable because it is also borrowed as mutable
    //~^ cannot borrow `f` as immutable because it is also borrowed as mutable
}

fn immut_plus_mut() {
    let mut f = Foo;

    &f
    +
    &mut f;  //~ ERROR: cannot borrow `f` as mutable because it is also borrowed as immutable
    //~^ cannot borrow `f` as mutable because it is also borrowed as immutable
}

fn main() {}
