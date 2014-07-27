// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::{Gc, GC};

fn borrow(x: &int, f: |x: &int|) {
    f(x)
}

fn test1(x: Gc<Box<int>>) {
    borrow(&*(*x).clone(), |p| {
        let x_a = &**x as *const int;
        assert!((x_a as uint) != (p as *const int as uint));
        assert_eq!(unsafe{*x_a}, *p);
    })
}

pub fn main() {
    test1(box(GC) box 22);
}
