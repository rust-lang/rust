// Test that attempt to swap `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/middle/borrowck/doc.rs

use std::util::swap;

fn foo<'a>(mut t0: &'a mut int,
           mut t1: &'a mut int) {
    let p: &int = &*t0;     // Freezes `*t0`
    swap(&mut t0, &mut t1); //~ ERROR cannot borrow `t0`
    *t1 = 22;
}

fn main() {
}
