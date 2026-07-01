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

// The `let` binding suggestion should still be offered alongside E0492.
fn deferred_assignment() {
    let r;
    r = &const { Cell::new(0) };
    //~^ ERROR interior mutable shared borrows of temporaries
    r.set(1);
}

// A `&mut` borrow is not a shared borrow, so it keeps the E0716 diagnostic.
fn mut_borrow() {
    let _: &'static mut _ = &mut const { Cell::new(0) };
    //~^ ERROR temporary value dropped while borrowed
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
