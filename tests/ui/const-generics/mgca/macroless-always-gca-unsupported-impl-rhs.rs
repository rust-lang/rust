#![feature(gca_min_const_items, gca_macroless_items)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize;
}

impl<const N: usize> Trait for [u8; N] {
    const ASSOC: usize = N + 1;
    //~^ ERROR the expression of this associated constant was unable to be treated as a gca constant
}

fn main() {}
