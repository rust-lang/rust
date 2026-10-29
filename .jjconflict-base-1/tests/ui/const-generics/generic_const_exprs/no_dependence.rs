//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn two_args<const N: usize, const M: usize>() -> [u8; M + 2] {
    [0; M + 2]
}

fn yay<const N: usize>() -> [u8; 4] {
     two_args::<N, 2>() // no lint
}

fn main() {}
