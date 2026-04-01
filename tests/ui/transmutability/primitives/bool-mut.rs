//@ check-fail

#![feature(transmutability)]
mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
    {}
}

fn main() {
    assert::is_transmutable::<&'static mut bool, &'static mut u8>() //~ ERROR cannot be safely transmuted
}
