//@ check-pass

//! The presence of safety invariants in the source type does not affect
//! transmutability.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src> // safety is NOT assumed
    {}
}

fn test() {
    #[repr(C)]
    struct Src;
    type Dst = ();
    assert::is_transmutable::<Src, Dst>();
}
