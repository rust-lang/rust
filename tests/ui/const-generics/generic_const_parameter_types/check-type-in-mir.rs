// Ensure that we actually treat `N`'s type as `&'a u32` in MIR typeck.

#![feature(unsized_const_params, adt_const_params, generic_const_parameter_types)]
//~^ WARN the feature `unsized_const_params` is incomplete
//~| WARN the feature `generic_const_parameter_types` is incomplete

fn foo<'a, const N: &'a u32>() {
    let b: &'static u32 = N;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
