// compile-flags: -Z borrowck=mir -Z two-phase-borrows

// This is similar to two-phase-reservation-sharing-interference.rs
// in that it shows a reservation that overlaps with a shared borrow.
//
// Currently, this test fails with lexical lifetimes, but succeeds
// with non-lexical lifetimes. (The reason is because the activation
// of the mutable borrow ends up overlapping with a lexically-scoped
// shared borrow; but a non-lexical shared borrow can end before the
// activation occurs.)
//
// So this test is just making a note of the current behavior.

#![feature(rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.push(shared.len());

    assert_eq!(v, [0, 1, 2, 3]);
}
