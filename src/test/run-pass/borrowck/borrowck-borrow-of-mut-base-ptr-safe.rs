// run-pass
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
// Test that freezing an `&mut` pointer while referent is
// frozen is legal.
//
// Example from src/librustc_borrowck/borrowck/README.md

// pretty-expanded FIXME #23616

fn foo<'a>(mut t0: &'a mut isize,
           mut t1: &'a mut isize) {
    let p: &isize = &*t0; // Freezes `*t0`
    let mut t2 = &t0;
    let q: &isize = &**t2; // Freezes `*t0`, but that's ok...
    let r: &isize = &*t0; // ...after all, could do same thing directly.
}

pub fn main() {
}
