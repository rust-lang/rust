//! Ensure that non exhaustive structs and enums with non exhaustive variants
//! aren't allowed to implement ConstParamTy under min_adt_const_params feature
//@ revisions: full min
//@[full] check-pass
#![cfg_attr(min, feature(min_adt_const_params))]
#![cfg_attr(full, feature(adt_const_params))]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[non_exhaustive]
#[derive(PartialEq, Eq, ConstParamTy)]
pub struct Miow;
   //[min]~^ ERROR: the trait `ConstParamTy` may not be implemented for this type

#[derive(PartialEq, Eq, ConstParamTy)]
pub enum Enumiow {
 //[min]~^ ERROR: the trait `ConstParamTy` may not be implemented for this type
    #[non_exhaustive] NonExhaustiveThingie,
    ExhaustiveThingie,
}

#[non_exhaustive]
#[derive(PartialEq, Eq, ConstParamTy)]
pub enum EnumiowButFine {
    ExhaustiveThingie,
    AlsoExhaustiveThingie,
}

fn main() {}
