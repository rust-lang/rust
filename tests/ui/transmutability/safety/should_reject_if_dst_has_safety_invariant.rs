//! Unless safety is assumed, a transmutation should be rejected if the
//! destination type may have a safety invariant.

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
    type Src = ();
    #[repr(C)]
    struct Dst;
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot be safely transmuted
}
