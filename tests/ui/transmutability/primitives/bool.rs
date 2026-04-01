//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(transmutability)]
mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
    {}

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY.and(Assume::VALIDITY) }>
    {}
}

fn main() {
    assert::is_transmutable::<u8, bool>(); //~ ERROR cannot be safely transmuted
    assert::is_maybe_transmutable::<u8, bool>();
    assert::is_transmutable::<bool, u8>();
}
