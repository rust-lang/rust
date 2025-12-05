//! Unless safety is assumed, a transmutation should be rejected if the
//! destination type may have a safety invariant.

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
    type Src = ();
    #[repr(C)]
    struct Dst;
    assert::is_transmutable::<Src, Dst>(); //~ ERROR cannot be safely transmuted
}
