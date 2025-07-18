// Regression test for #132882.

use std::ops::Add;

pub trait Numoid: Sized
where
    &'missing Self: Add<Self>,
    //~^ ERROR use of undeclared lifetime name `'missing`
{
}

// Proving `N: Numoid`'s well-formedness causes us to have to prove `&'missing N: Add<N>`.
// Since `'missing` is a region error, that will lead to us consider the predicate to hold,
// since it references errors. Since the freshener turns error regions into fresh regions,
// this means that subsequent lookups of `&'?0 N: Add<N>` will also hit this cache entry
// even if candidate assembly can't assemble anything for `&'?0 N: Add<?1>` anyways. This
// led to an ICE.
pub fn compute<N: Numoid>(a: N) {
    let _ = &a + a;
    //~^ ERROR cannot add `N` to `&N`
}

fn main() {}
