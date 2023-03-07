//! Validity may not be contracted, unless validity is assumed.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY }>
    {}
}

#[derive(Clone, Copy)] #[repr(u8)] enum Ox00 { V = 0x00 }
#[derive(Clone, Copy)] #[repr(u8)] enum Ox01 { V = 0x01 }
#[derive(Clone, Copy)] #[repr(u8)] enum OxFF { V = 0xFF }

fn test() {
    #[repr(C)]
    union Subset {
        a: Ox00,
        b: OxFF,
    }

    #[repr(C)]
    union Superset {
        a: Ox00,
        b: OxFF,
        c: Ox01,
    }

    assert::is_transmutable::<Superset, Subset>(); //~ ERROR cannot be safely transmuted
}
