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
    #[repr(C)] struct A(&'static B);
    #[repr(C)] struct B(&'static A);
    assert::is_maybe_transmutable::<&'static A, &'static B>();
    assert::is_maybe_transmutable::<&'static B, &'static A>();
}
