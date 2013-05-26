// Test that attempt to alias `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/middle/borrowck/doc.rs

use std::util::swap;

fn foo(t0: &mut int) {
    let p: &int = &*t0; // Freezes `*t0`
    let q: &const &mut int = &const t0; //~ ERROR cannot borrow `t0`
    **q = 22; //~ ERROR cannot assign to an `&mut` in a `&const` pointer
}

fn main() {
}