// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is just checking that we still reject code where temp values
// are borrowing values for longer than they will be around.
//
// Compare to run-pass/issue-23338-params-outlive-temps-of-body.rs

use std::cell::RefCell;

fn foo(x: RefCell<String>) -> String {
    let y = x;
    y.borrow().clone()
}
//~^ ERROR `y` does not live long enough

fn foo2(x: RefCell<String>) -> String {
    let ret = {
        let y = x;
        y.borrow().clone() //~ ERROR `y` does not live long enough
    };
    ret
}

fn main() {
    let r = RefCell::new(format!("data"));
    assert_eq!(foo(r), "data");
    let r = RefCell::new(format!("data"));
    assert_eq!(foo2(r), "data");
}
