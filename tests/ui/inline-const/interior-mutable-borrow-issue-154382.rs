// Verify that `&const { expr }` where `expr` has interior mutability
// gets E0492 instead of E0716.
// https://github.com/rust-lang/rust/issues/154382

use std::cell::Cell;
use std::sync::atomic::AtomicUsize;

struct Mutable(Cell<u32>);
impl Mutable {
    const fn new(a: u32) -> Self { Self(Cell::new(a)) }
}

fn foo() -> &'static Mutable {
    &const { Mutable::new(0) }
    //~^ ERROR interior mutable shared borrows of temporaries
}

struct Holder {
    val: &'static Mutable,
}

fn takes_static(_: &'static Mutable) {}

fn via_closure() {
    let _f: fn() -> &'static Mutable = || &const { Mutable::new(0) };
    //~^ ERROR interior mutable shared borrows of temporaries
}

fn via_struct() {
    let _h = Holder { val: &const { Mutable::new(0) } };
    //~^ ERROR interior mutable shared borrows of temporaries
}

fn via_argument() {
    takes_static(&const { Mutable::new(0) });
    //~^ ERROR interior mutable shared borrows of temporaries
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
