#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: usize>()
where
    [(); N + 1 + 1]:,
{
    bar();
    //~^ ERROR: type annotations
}

fn bar<const N: usize>()
where
    [(); N + 1]:,
{
}

fn main() {}
