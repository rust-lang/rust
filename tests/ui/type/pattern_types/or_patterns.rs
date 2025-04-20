//! Demonstrate some use cases of or patterns

//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"

#![feature(
    pattern_type_macro,
    pattern_types,
    rustc_attrs,
    const_trait_impl,
    pattern_type_range_trait
)]

use std::pat::pattern_type;

#[rustc_layout(debug)]
type NonNullI8 = pattern_type!(i8 is ..0 | 1..);
//~^ ERROR: layout_of

#[rustc_layout(debug)]
type NonNegOneI8 = pattern_type!(i8 is ..-1 | 0..);
//~^ ERROR: layout_of

fn main() {
    let _: NonNullI8 = 42;
    let _: NonNullI8 = 1;
    let _: NonNullI8 = 0;
    //~^ ERROR: mismatched types
    let _: NonNullI8 = -1;
    //~^ ERROR: cannot apply unary operator
    let _: NonNullI8 = -128;
    //~^ ERROR: cannot apply unary operator
    let _: NonNullI8 = 127;

    let _: NonNegOneI8 = 42;
    let _: NonNegOneI8 = 1;
    let _: NonNegOneI8 = 0;
    let _: NonNegOneI8 = -1;
    //~^ ERROR: cannot apply unary operator
    let _: NonNegOneI8 = -2;
    //~^ ERROR: cannot apply unary operator
    let _: NonNegOneI8 = -128;
    //~^ ERROR: cannot apply unary operator
    let _: NonNegOneI8 = 127;
}
