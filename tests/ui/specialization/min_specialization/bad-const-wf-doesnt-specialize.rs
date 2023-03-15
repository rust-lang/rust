#![feature(min_specialization)]

// An impl that has an erroneous const substitution should not specialize one
// that is well-formed.

struct S<const L: usize>;

impl<const N: i32> Copy for S<N> {}
impl<const M: usize> Copy for S<M> {}
//~^ ERROR conflicting implementations of trait `Copy` for type `S<_>`

fn main() {}
