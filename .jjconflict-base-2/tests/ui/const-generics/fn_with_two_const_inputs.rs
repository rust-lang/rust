#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const fn both(_: usize, b: usize) -> usize {
    b
}

fn foo<const N: usize, const M: usize>() -> [(); N + 2]
where
    [(); both(N + 1, M + 1)]:,
{
    bar()
    //~^ ERROR: unconstrained generic constant
}

fn bar<const N: usize>() -> [(); N]
where
    [(); N + 1]:,
{
    [(); N]
}

fn main() {}
