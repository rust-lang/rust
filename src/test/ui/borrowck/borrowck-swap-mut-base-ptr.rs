// Test that attempt to swap `&mut` pointer while pointee is borrowed
// yields an error.
//
// Example from src/librustc_borrowck/borrowck/README.md

use std::mem::swap;



fn foo<'a>(mut t0: &'a mut isize,
           mut t1: &'a mut isize) {
    let p: &isize = &*t0;     // Freezes `*t0`
    swap(&mut t0, &mut t1); //~ ERROR cannot borrow `t0`
    *t1 = 22;
    p.use_ref();
}

fn main() {
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
