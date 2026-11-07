//@ revisions: macroless macroful
//@[macroless] check-pass
//@ compile-flags: -Znext-solver

#![feature(
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    generic_const_items,
)]

#![cfg_attr(macroless, feature(gca_macroless_items))]

trait Trait {
    const ASSOC<const N: usize>: usize;
}

impl Trait for () {
    const ASSOC<const N: usize>: usize = N;
}

fn foo<const N: usize>() {
    let a: [(); <() as Trait>::ASSOC::<N>]
        = [(); N];
    //[macroful]~^ ERROR: mismatched types
}

fn main() {}
