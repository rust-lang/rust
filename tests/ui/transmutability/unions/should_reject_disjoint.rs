//! Validity must be satisfiable, even if validity is assumed.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY.and(Assume::VALIDITY) }>
    {}
}

#[derive(Clone, Copy)] #[repr(u8)] enum Ox00 { V = 0x00 }
#[derive(Clone, Copy)] #[repr(u8)] enum Ox01 { V = 0x01 }
#[derive(Clone, Copy)] #[repr(u8)] enum OxFF { V = 0xFF }

fn test() {
    #[repr(C)]
    union A {
        a: Ox00,
        b: OxFF,
    }

    #[repr(C)]
    union B {
        c: Ox01,
    }

    assert::is_maybe_transmutable::<A, B>(); //~ ERROR cannot be safely transmuted
    assert::is_maybe_transmutable::<B, A>(); //~ ERROR cannot be safely transmuted
}
