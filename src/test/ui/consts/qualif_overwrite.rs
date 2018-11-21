// compile-pass

#![feature(const_let)]

use std::cell::Cell;

const FOO: &Option<Cell<usize>> = {
    let mut a = Some(Cell::new(0));
    a = None; // resets `qualif(a)` to `qualif(None)`
    &{a}
};

fn main() {}
