// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that attempt to freeze an `&mut` pointer while referent is
// claimed yields an error.
//
// Example from src/middle/borrowck/doc.rs

fn foo<'a>(mut t0: &'a mut int,
           mut t1: &'a mut int) {
    let p: &mut int = &mut *t0; // Claims `*t0`
    let mut t2 = &t0;           //~ ERROR cannot borrow `t0`
    let q: &int = &**t2;        // Freezes `*t0` but not through `*p`
    *p += 1;                    // violates type of `*q`
}

fn main() {
}
