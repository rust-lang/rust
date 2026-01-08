#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]

fn takes_array<const A: [u32; 3]>() {}

fn generic_caller<const N: u32, const N2: u32>() {
    takes_array::<{ [N, N2, N] }>(); // ok

    takes_array::<{ [N, N2, const { 1 }] }>(); // ok

    takes_array::<{ [N, N2, const { 1 + 1 }] }>(); // ok

    takes_array::<{ [N, N2, 1] }>(); // ok

    takes_array::<{ [1, 1u32, 1_u32] }>(); // ok

    takes_array::<{ [N, N2, 1 + 1] }>(); // not implemented
    //~^ ERROR complex const arguments must be placed inside of a `const` block

    takes_array::<{ [N; 3] }>(); // not implemented
    //~^ ERROR complex const arguments must be placed inside of a `const` block
}

fn main() {}
