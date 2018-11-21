#![feature(const_let)]

use std::cell::Cell;

const FOO: &(Cell<usize>, bool) = {
    let mut a = (Cell::new(0), false);
    a.1 = true; // resets `qualif(a)` to `qualif(true)`
    &{a} //~ ERROR cannot borrow a constant which may contain interior mutability
};

fn main() {}