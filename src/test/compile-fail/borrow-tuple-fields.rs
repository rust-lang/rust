// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

struct Foo(Box<isize>, isize);

struct Bar(isize, isize);

fn main() {
    let x: (Box<_>, _) = (box 1, 2);
    let r = &x.0;
    let y = x; //~ ERROR cannot move out of `x` because it is borrowed

    let mut x = (1, 2);
    let a = &x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable because it is also borrowed as

    let mut x = (1, 2);
    let a = &mut x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable more than once at a time


    let x = Foo(box 1, 2);
    let r = &x.0;
    let y = x; //~ ERROR cannot move out of `x` because it is borrowed

    let mut x = Bar(1, 2);
    let a = &x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable because it is also borrowed as

    let mut x = Bar(1, 2);
    let a = &mut x.0;
    let b = &mut x.0; //~ ERROR cannot borrow `x.0` as mutable more than once at a time
}
