//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: usize>()
where
    [(); N + 1]:,
    [(); N + 1]:,
{
    bar::<N>();
}

fn bar<const N: usize>()
where
    [(); N + 1]:,
{
}

fn main() {}
