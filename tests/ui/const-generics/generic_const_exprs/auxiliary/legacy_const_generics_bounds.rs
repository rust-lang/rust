#![allow(internal_features)]
#![feature(rustc_attrs)]

// Signed Primitive Integers

#[rustc_legacy_const_generics(0)]
pub fn arg0_as_i8<const N: i8>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_i16<const N: i16>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_i32<const N: i32>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_i64<const N: i64>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_i128<const N: i128>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_isize<const N: isize>() {}

// Unsigned Primitive Integers

#[rustc_legacy_const_generics(0)]
pub fn arg0_as_u8<const N: u8>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_u16<const N: u16>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_u32<const N: u32>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_u64<const N: u64>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_u128<const N: u128>() {}
#[rustc_legacy_const_generics(0)]
pub fn arg0_as_usize<const N: usize>() {}
