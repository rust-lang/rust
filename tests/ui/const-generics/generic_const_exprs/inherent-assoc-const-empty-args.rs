// Regression test for <https://github.com/rust-lang/rust/issues/148063>

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Test<const N: usize>;

fn new<const N: usize>() -> Test<N>
where
    [(); N * 1]: Sized,
{
    Test
}

fn test<const N: usize>() -> Test<{ N - 1 }>
where
    [(); (N - usize::MAX) * 1]: Sized,
{
    new()
    //~^ ERROR mismatched types
}

fn main() {}
