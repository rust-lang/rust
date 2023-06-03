//@compile-flags: -Zmiri-tree-borrows

// Counterpart to tests/fail/tree-borrows/write-during-2phase.rs,
// this is the opposite situation: the Write is not problematic because
// the Protector has not yet been added and the Reserved has interior
// mutability.
use core::cell::Cell;

trait Thing: Sized {
    fn do_the_thing(&mut self, _s: i32) {}
}
impl<T> Thing for Cell<T> {}

fn main() {
    let mut x = Cell::new(1);
    let l = &x;

    x.do_the_thing({
        // Several Foreign accesses (both Reads and Writes) to the location
        // being reborrowed. Reserved + unprotected + interior mut
        // makes the pointer immune to everything as long as all accesses
        // are child accesses to its parent pointer x.
        x.set(3);
        l.set(4);
        x.get() + l.get()
    });
}
