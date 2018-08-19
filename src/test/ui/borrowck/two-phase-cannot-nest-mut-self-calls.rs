// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z two-phase-borrows

// This is the third counter-example from Niko's blog post
// smallcultfollowing.com/babysteps/blog/2017/03/01/nested-method-calls-via-two-phase-borrowing/
//
// It shows that not all nested method calls on `self` are magically
// allowed by this change. In particular, a nested `&mut` borrow is
// still disallowed.

fn main() {


    let mut vec = vec![0, 1];
    vec.get({

        vec.push(2);
        //~^ ERROR cannot borrow `vec` as mutable because it is also borrowed as immutable

        0
    });
}
