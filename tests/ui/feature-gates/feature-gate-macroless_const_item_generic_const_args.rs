//@ revisions: macroless macroful
//@[macroless] check-pass
//@ compile-flags: -Znext-solver

#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
)]

#![cfg_attr(macroless, feature(macroless_const_item_generic_const_args))]

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
