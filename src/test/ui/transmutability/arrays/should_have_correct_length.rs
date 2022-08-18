// check-pass
//! An array must have the correct length.

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

fn should_have_len_0() {
    type Array = [u8; 0];
    #[repr(C)] struct Struct();
    assert::is_maybe_transmutable::<Array, Struct>();
    assert::is_maybe_transmutable::<Struct, Array>();
}

fn should_have_len_1() {
    type Array = [u8; 1];
    #[repr(C)] struct Struct(u8);
    assert::is_maybe_transmutable::<Array, Struct>();
    assert::is_maybe_transmutable::<Struct, Array>();
}

fn should_have_len_2() {
    type Array = [u8; 2];
    #[repr(C)] struct Struct(u8, u8);
    assert::is_maybe_transmutable::<Array, Struct>();
    assert::is_maybe_transmutable::<Struct, Array>();
}

fn should_have_len_3() {
    type Array = [u8; 3];
    #[repr(C)] struct Struct(u8, u8, u8);
    assert::is_maybe_transmutable::<Array, Struct>();
    assert::is_maybe_transmutable::<Struct, Array>();
}
