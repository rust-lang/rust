#![feature(min_generic_const_args, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

trait Trait {
    type const ASSOC: usize;
}

fn takes_array<const A: [u32; 2]>() {}
fn takes_tuple_with_array<const A: ([u32; 2], u32)>() {}

fn generic_caller<T: Trait, const N: u32, const N2: u32>() {
    takes_array::<{ [N, N + 1] }>(); //~ ERROR complex const arguments must be placed inside of a `const` block
    takes_tuple_with_array::<{ ([N, N + 1], N) }>(); //~ ERROR complex const arguments must be placed inside of a `const` block
}

fn main() {}
