#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: usize, const M: usize>() -> [(); N+2]
where
    [(); N + 1]:,
    [(); M + 1]:,
{
    bar()
    //~^ ERROR: unconstrained
}

fn bar<const N: usize>() -> [(); N]
where
    [(); N + 1]:,
{
    [(); N]
}

fn main() {}
