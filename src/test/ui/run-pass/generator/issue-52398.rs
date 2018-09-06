// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators)]

use std::cell::RefCell;

struct A;

impl A {
    fn test(&self, a: ()) {}
}

fn main() {
    // Test that the MIR local with type &A created for the auto-borrow adjustment
    // is caught by typeck
    move || {
        A.test(yield);
    };

    // Test that the std::cell::Ref temporary returned from the `borrow` call
    // is caught by typeck
    let y = RefCell::new(true);
    static move || {
        yield *y.borrow();
        return "Done";
    };
}
