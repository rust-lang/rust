// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #48551. Covers a case where duplicate candidates
// arose during associated type projection.

use std::ops::{Mul, MulAssign};

pub trait ClosedMul<Right>: Sized + Mul<Right, Output = Self> + MulAssign<Right> {}
impl<T, Right> ClosedMul<Right> for T
where
    T: Mul<Right, Output = T> + MulAssign<Right>,
{
}

pub trait InnerSpace: ClosedMul<<Self as InnerSpace>::Real> {
    type Real;
}

pub trait FiniteDimVectorSpace: ClosedMul<<Self as FiniteDimVectorSpace>::Field> {
    type Field;
}

pub trait FiniteDimInnerSpace
    : InnerSpace + FiniteDimVectorSpace<Field = <Self as InnerSpace>::Real> {
}

pub trait EuclideanSpace: ClosedMul<<Self as EuclideanSpace>::Real> {
    type Coordinates: FiniteDimInnerSpace<Real = Self::Real>
        + Mul<Self::Real, Output = Self::Coordinates>
        + MulAssign<Self::Real>;

    type Real;
}

fn main() {}
