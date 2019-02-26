use std::cell::Cell;

// this is overly conservative. The reset to `None` should clear `a` of all qualifications
// while we could fix this, it would be inconsistent with `qualif_overwrite_2.rs`.
// We can fix this properly in the future by allowing constants that do not depend on generics
// to be checked by an analysis on the final value instead of looking at the body.
const FOO: &Option<Cell<usize>> = {
    let mut a = Some(Cell::new(0));
    a = None; // sets `qualif(a)` to `qualif(a) | qualif(None)`
    &{a} //~ ERROR cannot borrow a constant which may contain interior mutability
};

fn main() {}
