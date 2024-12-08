//! ALL valid bit patterns of the source must be valid bit patterns of the
//! destination type, unless validity is assumed.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
        // validity is NOT assumed -----^^^^^^^^^^^^^^^^^^
    {}
}

#[derive(Clone, Copy)] #[repr(u8)] enum Ox00 { V = 0x00 }
#[derive(Clone, Copy)] #[repr(u8)] enum Ox7F { V = 0x7F }
#[derive(Clone, Copy)] #[repr(u8)] enum OxFF { V = 0xFF }

fn test() {
    #[repr(C)]
    union A {
        a: Ox00,
        b: Ox7F,
    }

    #[repr(C)]
    union B {
        a: Ox7F,
        b: OxFF,
    }

    assert::is_transmutable::<A, B>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<B, A>(); //~ ERROR cannot be safely transmuted
}
