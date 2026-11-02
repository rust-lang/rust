//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: true,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}
}

fn main() {
    #[repr(C)] struct A(bool, &'static A);
    #[repr(C)] struct B(u8, &'static B);
    assert::is_maybe_transmutable::<&'static A, &'static B>();
}
