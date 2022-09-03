// check-pass
//! If validity is assumed, there need only be one matching bit-pattern between
//! the source and destination types.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY.and(Assume::VALIDITY) }>
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

    assert::is_maybe_transmutable::<A, B>();
    assert::is_maybe_transmutable::<B, A>();
}
