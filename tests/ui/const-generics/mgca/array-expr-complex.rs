//@ revisions: r1 r2 r3

#![expect(incomplete_features)]
#![feature(gca_min_const_items, adt_const_params)]

use std::gca;

fn takes_array<const A: [u32; 3]>() {}

fn generic_caller<const X: u32, const Y: usize>() {
    // not supported yet
    #[cfg(r1)]
    takes_array::<{ gca!([1, 2, 1 + 2]) }>();
    //[r1]~^ ERROR: complex const arguments must be placed inside of a `const` block
    #[cfg(r2)]
    takes_array::<{ gca!([X; 3]) }>();
    //[r2]~^ ERROR: complex const arguments must be placed inside of a `const` block
    #[cfg(r3)]
    takes_array::<{ gca!([0; Y]) }>();
    //[r3]~^ ERROR: complex const arguments must be placed inside of a `const` block
}

fn main() {}
