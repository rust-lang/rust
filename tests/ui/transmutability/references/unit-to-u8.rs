//@ check-fail
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, {
            Assume {
                alignment: true,
                lifetimes: true,
                safety: true,
                validity: false,
            }
        }>
    {}
}

fn main() {
    #[repr(C)] struct Unit;
    assert::is_maybe_transmutable::<&'static Unit, &'static u8>(); //~ ERROR cannot be safely transmuted
}
