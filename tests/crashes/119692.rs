//@ known-bug: #119692
//@ compile-flags: -Copt-level=0
#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

use std::ops::Add;

#[derive(PartialEq, Eq, Clone, Debug, core::marker::ConstParamTy)]
pub struct Dimension;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Default)]
pub struct Quantity<S, const D: Dimension>(pub(crate) S);

impl<const D: Dimension, LHS, RHS> Add<Quantity<RHS, D>> for Quantity<LHS, D>
where
    LHS: Add<RHS>,
{
    type Output = Quantity<<LHS as Add<RHS>>::Output, D>;
    fn add(self, rhs: Quantity<RHS, D>) -> Self::Output {
        Quantity(self.0 + rhs.0)
    }
}

impl<LHS, RHS> Add<RHS> for Quantity<LHS, { Dimension }>
where
    LHS: Add<RHS>,
{
    type Output = Quantity<<LHS as Add<RHS>>::Output, { Dimension }>;
    fn add(self, rhs: RHS) -> Self::Output {
        Quantity(self.0 + rhs)
    }
}

impl Add<Quantity<f32, { Dimension }>> for f32 {
    type Output = Quantity<f32, { Dimension }>;
    fn add(self, rhs: Quantity<f32, { Dimension }>) -> Self::Output {
        Quantity(self + rhs.0)
    }
}

pub fn add<const U: Dimension>(x: Quantity<f32, U>, y: Quantity<f32, U>) -> Quantity<f32, U> {
    x + y
}

fn main() {
    add(Quantity::<f32, {Dimension}>(1.0), Quantity(2.0));
}
