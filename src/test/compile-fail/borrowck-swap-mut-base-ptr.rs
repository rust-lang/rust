// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that attempt to swap `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/middle/borrowck/doc.rs

use std::mem::swap;

fn foo<'a>(mut t0: &'a mut int,
           mut t1: &'a mut int) {
    let p: &int = &*t0;     // Freezes `*t0`
    swap(&mut t0, &mut t1); //~ ERROR cannot borrow `t0`
    *t1 = 22;
}

fn main() {
}
