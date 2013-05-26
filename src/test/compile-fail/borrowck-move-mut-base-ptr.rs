// Test that attempt to move `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/middle/borrowck/doc.rs

use std::util::swap;

fn foo(t0: &mut int) {
    let p: &int = &*t0; // Freezes `*t0`
    let t1 = t0;        //~ ERROR cannot move out of `t0`
    *t1 = 22;
}

fn main() {
}