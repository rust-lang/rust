// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is largely checking that we now accept code where temp values
// are borrowing from the input parameters (the `foo` case below).
//
// Compare to run-pass/issue-23338-params-outlive-temps-of-body.rs
//
// (The `foo2` case is just for parity with the above test, which
//  shows what happens when you move the `y`-binding to the inside of
//  the inner block.)

use std::cell::RefCell;

fn foo(x: RefCell<String>) -> String {
    x.borrow().clone()
}

fn foo2(x: RefCell<String>) -> String {
    let y = x;
    let ret = {
        y.borrow().clone()
    };
    ret
}

pub fn main() {
    let r = RefCell::new(format!("data"));
    assert_eq!(foo(r), "data");
    let r = RefCell::new(format!("data"));
    assert_eq!(foo2(r), "data");
}
