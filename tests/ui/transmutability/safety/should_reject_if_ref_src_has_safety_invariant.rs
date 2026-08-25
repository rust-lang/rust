//! Unless safety is assumed, a transmutation over exclusive references should
//! be rejected if the source potentially carries safety invariants.

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
    struct Src {
        non_zero: u8,
    }
    type Dst = u8;
    assert::is_transmutable::<&mut Src, &mut Dst>(); //~ ERROR cannot be safely transmuted
}
