#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(incomplete_features)]

struct S;

impl S {
    const N: usize = 42;
}

fn main() {
    let _x: [(); S::N] = todo!();
    //~^ ERROR inherent associated types are unstable
}
