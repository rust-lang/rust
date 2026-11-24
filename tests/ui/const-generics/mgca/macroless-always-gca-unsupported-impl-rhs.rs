#![feature(gca_min_const_items, gca_macroless_items)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize;
}

impl<const N: usize> Trait for [u8; N] {
    const ASSOC: usize = N + 1;
    //~^ ERROR implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
}

fn main() {}
