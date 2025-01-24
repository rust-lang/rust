//@ check-pass

//! When safety is assumed, a transmutation over exclusive references should be
//! accepted if the source type potentially carries safety invariants.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
    {}
}

fn test() {
    #[repr(C)]
    struct Src {
        non_zero: u8,
    }
    type Dst = u8;
    assert::is_transmutable::<&mut Src, &mut Dst>();
}
