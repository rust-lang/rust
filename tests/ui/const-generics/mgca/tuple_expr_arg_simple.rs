//@ check-pass

#![feature(min_generic_const_args, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

trait Trait {
    type const ASSOC: u32;
}

fn takes_tuple<const A: (u32, u32)>() {}
fn takes_nested_tuple<const A: (u32, (u32, u32))>() {}

fn generic_caller<T: Trait, const N: u32, const N2: u32>() {
    takes_tuple::<{ core::direct_const_arg!((N, N2)) }>();
    takes_tuple::<{ core::direct_const_arg!((N, T::ASSOC)) }>();

    takes_nested_tuple::<{ core::direct_const_arg!((N, (N, N2))) }>();
    takes_nested_tuple::<{ core::direct_const_arg!((N, (N, T::ASSOC))) }>();
}

fn main() {}
