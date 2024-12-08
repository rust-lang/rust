#![feature(generic_const_exprs)]
//~^ WARN: the feature `generic_const_exprs` is incomplete

// Regression test for #125770 which would ICE under the old effects desugaring that
// created a const generic parameter for constness on `Add`.

use std::ops::Add;

pub struct Dimension;

pub struct Quantity<S, const D: Dimension>(S);
//~^ ERROR: `Dimension` is forbidden as the type of a const generic parameter

impl<const D: Dimension, LHS, RHS> Add<LHS, D> for Quantity<LHS, { Dimension }> {}
//~^ ERROR: trait takes at most 1 generic argument
//~| ERROR: `Dimension` is forbidden as the type of a const generic parameter

pub fn add<const U: Dimension>(x: Quantity<f32, U>) -> Quantity<f32, U> {
    //~^ ERROR: `Dimension` is forbidden as the type of a const generic parameter
    x + y
    //~^ ERROR: cannot find value `y` in this scope
}

fn main() {}
