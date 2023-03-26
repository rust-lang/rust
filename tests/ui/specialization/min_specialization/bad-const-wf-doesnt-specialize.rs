#![feature(min_specialization)]

// An impl that has an erroneous const substitution should not specialize one
// that is well-formed.
#[derive(Clone)]
struct S<const L: usize>;

impl<const N: i32> Copy for S<N> {}
//~^ ERROR the constant `N` is not of type `usize`
impl<const M: usize> Copy for S<M> {}

fn main() {}
