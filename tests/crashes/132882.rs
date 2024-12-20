//@ known-bug: #132882

use std::ops::Add;

pub trait Numoid
where
    for<N: Numoid> &'a Self: Add<Self>,
{
}

pub fn compute<N: Numoid>(a: N) -> N {
    &a + a
}
