//@ run-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]
#![allow(dead_code)]

fn takes_array<const A: [u32; 3]>() {}

trait Trait {
    #[type_const]
    const ASSOC: u32;
}

fn generic_caller<T: Trait, const N: u32>() {
    takes_array::<{ [T::ASSOC, N, T::ASSOC] }>();
    takes_array::<{ [1_u32, T::ASSOC, 2] }>();
}

fn main() {}
