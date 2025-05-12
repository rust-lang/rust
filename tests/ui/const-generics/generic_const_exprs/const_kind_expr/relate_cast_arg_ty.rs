// checks that when we relate a `Expr::Cast` we also relate the type of the
// const argument.
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: usize>() -> [(); (true as usize) + N] {
    [(); (1_u8 as usize) + N]
    //~^ error: mismatched types
    //~| error: unconstrained generic constant
}

fn main() {}
