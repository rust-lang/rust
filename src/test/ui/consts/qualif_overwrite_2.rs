#![feature(const_refs_to_cell)]

use std::cell::Cell;

// const qualification is not smart enough to know about fields and always assumes that there might
// be other fields that caused the qualification
const FOO: &Option<Cell<usize>> = { //~ ERROR may contain interior mutability
    let mut a = (Some(Cell::new(0)),);
    a.0 = None; // sets `qualif(a)` to `qualif(a) | qualif(None)`
    &{a.0}
};

fn main() {}
