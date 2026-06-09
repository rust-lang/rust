//@ revisions: r1 r2 r3

#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]

fn takes_array<const A: [u32; 3]>() {}

fn generic_caller<const X: u32, const Y: usize>() {
    // not supported yet
    #[cfg(r1)]
    takes_array::<{ [1, 2, 1 + 2] }>();
    //[r1]~^ ERROR: complex const arguments must be placed inside of a `const` block
    #[cfg(r2)]
    takes_array::<{ [X; 3] }>();
    //[r2]~^ ERROR: complex const arguments must be placed inside of a `const` block
    #[cfg(r3)]
    takes_array::<{ [0; Y] }>();
    //[r3]~^ ERROR: complex const arguments must be placed inside of a `const` block
}

fn main() {}
