// Regression test for #92230.
//
// check-pass

#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub trait Super {}
pub trait Sub: Super {}

impl<A> const Super for &A where A: ~const Super {}
impl<A> const Sub for &A where A: ~const Sub {}

fn main() {}
