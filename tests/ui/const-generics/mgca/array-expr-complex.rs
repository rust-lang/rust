#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]

fn takes_array<const A: [u32; 3]>() {}

fn generic_caller<const X: u32, const Y: usize>() {
    // not supported yet
    takes_array::<{ [1, 2, 1 + 2] }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    takes_array::<{ [X; 3] }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    takes_array::<{ [0; Y] }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
}

fn main() {}
