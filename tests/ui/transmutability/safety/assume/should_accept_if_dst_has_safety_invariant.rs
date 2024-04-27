//@ check-pass

//! When safety is assumed, a transmutation should be accepted if the
//! destination type might carry a safety invariant.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, { Assume::SAFETY }>
    {}
}

fn test() {
    type Src = ();
    #[repr(C)]
    struct Dst;
    assert::is_transmutable::<Src, Dst>();
}
