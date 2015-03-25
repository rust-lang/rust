// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #23698: The reassignment checker only cared
// about the last assigment in a match arm body

// Use an extra function to make sure no extra assignments
// are introduced by macros in the match statement
fn check_eq(x: i32, y: i32) {
    assert_eq!(x, y);
}

#[allow(unused_assignments)]
fn main() {
    let mut x = Box::new(1);
    match x {
        y => {
            x = Box::new(2);
            let _tmp = 1; // This assignment used to throw off the reassignment checker
            check_eq(*y, 1);
        }
    }
}
