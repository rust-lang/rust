#![feature(const_refs_to_cell)]

use std::cell::Cell;

const FOO: &(Cell<usize>, bool) = { //~ ERROR may contain interior mutability
    let mut a = (Cell::new(0), false);
    a.1 = true; // sets `qualif(a)` to `qualif(a) | qualif(true)`
    &{a}
};

fn main() {}
