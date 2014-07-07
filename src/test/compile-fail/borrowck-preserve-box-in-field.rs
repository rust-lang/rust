// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

#![feature(managed_boxes)]

use std::gc::GC;

fn borrow(x: &int, f: |x: &int|) {
    let before = *x;
    f(x);
    let after = *x;
    assert_eq!(before, after);
}

struct F { f: Box<int> }

pub fn main() {
    let mut x = box(GC) F {f: box 3};
    borrow(&*x.f, |b_x| {
    //~^ ERROR cannot borrow `x` as mutable because `*x.f` is also borrowed as immutable
        assert_eq!(*b_x, 3);
        assert_eq!(&(*x.f) as *const int, &(*b_x) as *const int);
        //~^ NOTE borrow occurs due to use of `x` in closure
        x = box(GC) F {f: box 4};

        println!("&*b_x = {:p}", &(*b_x));
        assert_eq!(*b_x, 3);
        assert!(&(*x.f) as *const int != &(*b_x) as *const int);
    })
}
