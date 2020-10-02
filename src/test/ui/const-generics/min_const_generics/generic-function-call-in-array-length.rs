#![feature(min_const_generics)]

const fn foo(n: usize) -> usize { n * 2 }

fn bar<const N: usize>() -> [u32; foo(N)] {
    //~^ ERROR generic parameters must not be used inside of non-trivial constant values
    [0; foo(N)]
    //~^ ERROR generic parameters must not be used inside of non-trivial constant values
}

fn main() {}
