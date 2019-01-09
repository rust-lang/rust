// Regression test for #25954: detect and reject a closure type that
// references itself.

use std::cell::{Cell, RefCell};

struct A<T: Fn()> {
    x: RefCell<Option<T>>,
    b: Cell<i32>,
}

fn main() {
    let mut p = A{x: RefCell::new(None), b: Cell::new(4i32)};

    // This is an error about types of infinite size:
    let q = || p.b.set(5i32); //~ ERROR mismatched types

    *(p.x.borrow_mut()) = Some(q);
}
