// check-fail
#![allow(incomplete_features)]
#![feature(impl_exhaustive_const_traits)]
#![feature(generic_const_exprs)]

trait Dot {}

struct Mat<const N: usize>;

impl Dot for Mat<0> {}
impl<const N: usize> Dot for Mat<N> where Mat<{N-1}>: Dot {}
//~^ ERROR: evaluation of

fn main() {}
