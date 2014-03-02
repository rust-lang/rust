// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A(uint);

impl A {
    fn a(self@A(u)) -> uint {
        u
    }

    fn b(&mut self@&A(ref mut u), i: uint) {
        *u = i;
    }

    fn c(~self@~A(u)) -> uint {
        u
    }
}

struct B<T> {
    t: T,
}

impl<T> B<T> {
    fn a(self@B { t: t }) -> T {
        t
    }

    fn b(&mut self@&B { t: ref mut t }, i: T) {
        *t = i;
    }
}

fn main() {
    let a = A(1);
    assert_eq!(a.a(), 1);
    let mut a = A(2);
    a.b(3);
    assert_eq!(a.a(), 3);
    // FIXME A::c crashes (probably related to #12534)
    // let a = ~A(1);
    // assert_eq!(a.c(), 1);

    let b = B { t: false };
    assert!(!b.a());
    let mut b = B { t: 0 as uint };
    b.b(3);
    assert_eq!(b.a(), 3);
}
