//! Test that we don't ICE when passing the wrong ADT to ASSUME.

#![feature(adt_const_params)]
#![feature(transmutability)]

use std::marker::ConstParamTy;
use std::mem::TransmuteFrom;

#[derive(ConstParamTy, PartialEq, Eq)]
struct NotAssume;

fn foo<const ASSUME: NotAssume>()
where
    u8: TransmuteFrom<u8, ASSUME>, //~ ERROR the constant `ASSUME` is not of type `Assume`
{
}

fn main() {
    foo::<{ NotAssume }>();
}
