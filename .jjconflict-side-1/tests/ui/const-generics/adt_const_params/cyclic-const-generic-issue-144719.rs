//! Const generic variant of #144719

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Thing(&'static Thing);

static X: Thing = Thing(&X);
const Y: &Thing = &X;

fn foo<const N: &'static Thing>() -> usize { 0 }

fn main() {
    foo::<Y>();
    //~^ ERROR constant main::{constant#0} cannot be used as pattern
    //~| ERROR constant main::{constant#0} cannot be used as pattern
}
