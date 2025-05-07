//@ check-fail
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: true,
                safety: true,
                validity: true,
            }
        }>
    {}
}

fn main() {
    assert::is_maybe_transmutable::<&'static [u8; 0], &'static [u16; 0]>(); //~ ERROR `&[u8; 0]` cannot be safely transmuted into `&[u16; 0]`
}
