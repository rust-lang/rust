// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

fn f(mut y: Box<int>) {
    *y = 5;
    assert_eq!(*y, 5);
}

fn g() {
    let frob = |&: mut q: Box<int>| { *q = 2; assert!(*q == 2); };
    let w = box 37;
    frob(w);

}

pub fn main() {
    let z = box 17;
    f(z);
    g();
}
