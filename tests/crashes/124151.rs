//@ known-bug: #124151
#![feature(generic_const_exprs)]

use std::ops::Add;

pub struct Dimension;

pub struct Quantity<S, const D: Dimension>(S);

impl<const D: Dimension, LHS, RHS> Add<LHS, D> for Quantity<LHS, { Dimension }> {}

pub fn add<const U: Dimension>(x: Quantity<f32, U>) -> Quantity<f32, U> {
    x + y
}
