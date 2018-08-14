// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

// Test that attempt to move `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/librustc_borrowck/borrowck/README.md

fn foo(t0: &mut isize) {
    let p: &isize = &*t0; // Freezes `*t0`
    let t1 = t0;        //~ ERROR cannot move out of `t0`
    *t1 = 22;
}

fn main() {
}
