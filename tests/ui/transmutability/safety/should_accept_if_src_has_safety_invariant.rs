//@ check-pass

//! The presence of safety invariants in the source type does not affect
//! transmutability.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src> // safety is NOT assumed
    {}
}

fn test() {
    #[repr(C)]
    struct Src;
    type Dst = ();
    assert::is_transmutable::<Src, Dst>();
}
