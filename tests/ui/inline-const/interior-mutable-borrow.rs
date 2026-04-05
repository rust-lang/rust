// Verify that `&const { expr }` where `expr` has interior mutability
// gets E0492 instead of E0716.

use std::cell::Cell;
use std::sync::atomic::AtomicUsize;

struct Mutable(Cell<u32>);
impl Mutable {
    const fn new(a: u32) -> Self { Self(Cell::new(a)) }
}

fn main() {
    let _: &'static _ = &const { 0u32 };

    let _: &'static _ = &const { Mutable::new(0u32) };
    //~^ ERROR interior mutable shared borrows of temporaries

    let _: &'static _ = const { &Mutable::new(0u32) };
    //~^ ERROR interior mutable shared borrows of temporaries

    let _: &'static _ = &const { AtomicUsize::new(0) };
    //~^ ERROR interior mutable shared borrows of temporaries
}
