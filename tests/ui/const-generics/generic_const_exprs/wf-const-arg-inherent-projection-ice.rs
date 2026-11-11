//! Regression test for <https://github.com/rust-lang/rust/issues/159561>.

#![feature(generic_const_exprs)]

fn new<const N: usize>()
where
    [(); N * 1]:,
{
}

fn test<const N: usize>()
where
    [(); N - usize::MAX * 1]:,
{
    new
    //~^ ERROR mismatched types
}

fn main() {}
