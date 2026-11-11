//@ check-fail
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
    assert::is_maybe_transmutable::<&'static A, &'static mut B>(); //~ ERROR cannot be safely transmuted
}
