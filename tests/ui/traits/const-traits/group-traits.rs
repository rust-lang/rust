//@ check-pass
use std::ops::Add;

pub trait GroupOpsOwned<Rhs = Self, Output = Self>: for<'r> Add<&'r Rhs, Output = Output> {}

pub trait Curve: Sized + GroupOpsOwned<Self::AffineRepr> {
    type AffineRepr;
}

pub trait CofactorCurve: Curve<AffineRepr = <Self as CofactorCurve>::Affine> {
    type Affine;
}

fn main() {}
