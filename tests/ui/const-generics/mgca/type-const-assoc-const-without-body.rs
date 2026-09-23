//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

trait Tr {
    #[rustc_always_gca]
    const SIZE: usize;
}

struct T;

impl Tr for T {
    const SIZE: usize;
    //~^ ERROR associated constant in `impl` without body
    //~| ERROR implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
}

fn main() {}
