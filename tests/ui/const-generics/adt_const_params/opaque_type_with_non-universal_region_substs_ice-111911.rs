//@ edition:2021
//@ check-pass
// issues rust-lang/rust#111911
// test for ICE opaque type with non-universal region substs

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

pub async fn foo<const X: &'static str>() {}
fn bar<const N: &'static u8>() -> impl Sized {}

pub fn main() {}
