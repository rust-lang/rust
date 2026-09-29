//@ check-pass

#![feature(gca_min_const_items, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const ASSOC: u32;
}

fn takes_tuple<const A: (u32, u32)>() {}
fn takes_nested_tuple<const A: (u32, (u32, u32))>() {}

fn generic_caller<T: Trait, const N: u32, const N2: u32>() {
    takes_tuple::<{ gca!((N, N2)) }>();
    takes_tuple::<{ gca!((N, T::ASSOC)) }>();

    takes_nested_tuple::<{ gca!((N, (N, N2))) }>();
    takes_nested_tuple::<{ gca!((N, (N, T::ASSOC))) }>();
}

fn main() {}
