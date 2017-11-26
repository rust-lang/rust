// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn foo(x: Box<isize>) -> isize {
    let y = &*x;
    free(x); //[ast]~ ERROR cannot move out of `x` because it is borrowed
    //[mir]~^ ERROR cannot move out of `x` because it is borrowed
    *y
}

fn free(_x: Box<isize>) {
}

fn main() {
}
