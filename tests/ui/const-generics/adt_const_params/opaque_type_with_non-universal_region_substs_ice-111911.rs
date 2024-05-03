//@ edition:2021
// issues rust-lang/rust#111911
// test for ICE opaque type with non-universal region substs

#![feature(adt_const_params)]
#![allow(incomplete_features)]

pub async fn foo<const X: &'static str>() {}
//~^ ERROR const parameter `X` is part of concrete type but not used in parameter list for the `impl Trait` type alias
//~| ERROR const parameter `X` is part of concrete type but not used in parameter list for the `impl Trait` type alias
fn bar<const N: &'static u8>() -> impl Sized {}

pub fn main() {}
