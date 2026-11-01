// Ensure that we actually treat `N`'s type as `Invariant<'static>` in MIR typeck.

#![feature(adt_const_params)]

use std::marker::ConstParamTy;
use std::ops::Deref;

#[derive(ConstParamTy, PartialEq, Eq)]
struct Invariant<'a>(<&'a () as Deref>::Target);

fn test<'a, const N: Invariant<'static>>() {
    let x: Invariant<'a> = N;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
