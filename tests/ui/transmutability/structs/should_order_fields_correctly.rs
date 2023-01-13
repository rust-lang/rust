// check-pass
//! The fields of a struct should be laid out in lexical order.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
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

#[repr(C)] struct S01(V0, V1);
#[repr(C)] struct S012(V0, V1, V2);

fn should_order_tag_and_fields_correctly() {
    // An implementation that (incorrectly) arranges S01 as [0x01, 0x00] will,
    // in principle, reject this transmutation.
    assert::is_transmutable::<S01, V0>();
    // Again, but with one more field.
    assert::is_transmutable::<S012, S01>();
}
