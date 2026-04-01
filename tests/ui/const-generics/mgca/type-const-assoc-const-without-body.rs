//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Tr {
    type const SIZE: usize;
}

struct T;

impl Tr for T {
    type const SIZE: usize;
    //~^ ERROR associated constant in `impl` without body
}

fn main() {}
