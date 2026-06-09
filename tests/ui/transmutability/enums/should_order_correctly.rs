//@ check-pass
//! The payloads of an enum variant should be ordered after its tag.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume::ALIGNMENT
                .and(Assume::LIFETIMES)
                .and(Assume::SAFETY)
                .and(Assume::VALIDITY)
        }>
    {}
}

#[repr(u8)] enum V0 { V = 0 }
#[repr(u8)] enum V1 { V = 1 }
#[repr(u8)] enum V2 { V = 2 }

#[repr(u8)] enum E01 { V0(V1) = 0u8 }
#[repr(u8)] enum E012 { V0(V1, V2) = 0u8 }

fn should_order_tag_and_fields_correctly() {
    // An implementation that (incorrectly) arranges E01 as [0x01, 0x00] will,
    // in principle, reject this transmutation.
    assert::is_transmutable::<E01, V0>();
    // Again, but with one more field.
    assert::is_transmutable::<E012, E01>();
}
