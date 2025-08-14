use std::cell::Cell;

const FOO: &(Cell<usize>, bool) = {
    let mut a = (Cell::new(0), false);
    a.1 = true; // sets `qualif(a)` to `qualif(a) | qualif(true)`
    &{a} //~ ERROR cannot refer to interior mutable
};

fn main() {}
