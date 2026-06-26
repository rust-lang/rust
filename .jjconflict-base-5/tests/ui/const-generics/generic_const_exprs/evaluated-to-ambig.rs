//@ check-pass

// We previously always returned ambiguity when equating generic consts, even if they
// only contain generic parameters. This is incorrect as trying to unify `N > 1` with `M > 1`
// should fail.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

enum Assert<const COND: bool> {}
trait IsTrue {}
impl IsTrue for Assert<true> {}

struct Foo<const N: usize, const M: usize>;
trait Bar<const N: usize, const M: usize> {}
impl<const N: usize, const M: usize> Bar<N, M> for Foo<N, M>
where
    Assert<{ N > 1 }>: IsTrue,
    Assert<{ M > 1 }>: IsTrue,
{
}

fn main() {}
