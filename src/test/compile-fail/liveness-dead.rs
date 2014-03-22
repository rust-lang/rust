// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![deny(dead_assignment)]

fn f1(x: &mut int) {
    *x = 1; // no error
}

fn f2() {
    let mut x: int = 3; //~ ERROR: value assigned to `x` is never read
    x = 4;
    x.clone();
}

fn f3() {
    let mut x: int = 3;
    x.clone();
    x = 4; //~ ERROR: value assigned to `x` is never read
}

fn main() {}
