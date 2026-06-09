// Test that the NLL `relate_tys` code correctly deduces that a
// function returning always its first argument can be upcast to one
// that returns either first or second argument.

use std::cell::Cell;

type DoubleCell<A> = Cell<(A, A)>;
type DoublePair<A> = (A, A);

fn make_cell<'b>(x: &'b u32) -> Cell<(&'static u32, &'b u32)> {
    panic!()
}

fn main() {
    let a: &'static u32 = &22;
    let b = 44;

    // Here we get an error because `DoubleCell<_>` requires the same type
    // on both parts of the `Cell`, and we can't have that.
    let x: DoubleCell<_> = make_cell(&b); //~ ERROR

    // Here we do not get an error because `DoublePair<_>` permits
    // variance on the lifetimes involved.
    let y: DoublePair<_> = make_cell(&b).get();
}
