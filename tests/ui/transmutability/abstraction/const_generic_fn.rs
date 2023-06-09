// check-pass
//! An array must have the correct length.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn array_like<T, E, const N: usize>()
    where
        T: BikeshedIntrinsicFrom<[E; N], Context, { Assume::SAFETY }>,
        [E; N]: BikeshedIntrinsicFrom<T, Context, { Assume::SAFETY }>
    {}
}

fn len_0() {
    type Array = [u8; 0];
    #[repr(C)] struct Struct();
    assert::array_like::<Struct, u8, 0>();
}

fn len_1() {
    type Array = [u8; 1];
    #[repr(C)] struct Struct(u8);
    assert::array_like::<Struct, u8, 1>();
}

fn len_2() {
    type Array = [u8; 2];
    #[repr(C)] struct Struct(u8, u8);
    assert::array_like::<Struct, u8, 2>();
}

fn len_3() {
    type Array = [u8; 3];
    #[repr(C)] struct Struct(u8, u8, u8);
    assert::array_like::<Struct, u8, 3>();
}
