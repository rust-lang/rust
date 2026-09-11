//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Tr {
    #[rustc_always_gca]
    const SIZE: usize;
}

struct T;

impl Tr for T {
    const SIZE: usize;
    //~^ ERROR associated constant in `impl` without body
    //~| ERROR implementation of a `#[rustc_always_gca]` must have a `direct_const_arg!` RHS
}

fn main() {}
