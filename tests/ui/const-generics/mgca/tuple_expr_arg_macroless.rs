//@ check-pass

#![feature(
    adt_const_params,
    gca_adts,
    gca_macroless_args,
    gca_min_const_items,
    unsized_const_params
)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: u32;
}

fn takes_tuple<const A: (u32, u32)>() {}
fn takes_nested_tuple<const A: (u32, (u32, u32))>() {}

fn generic_caller<T: Trait, const N: u32, const N2: u32>() {
    takes_tuple::<{ (N, N2) }>();
    takes_tuple::<{ (N, T::ASSOC) }>();

    takes_nested_tuple::<{ (N, (N, N2)) }>();
    takes_nested_tuple::<{ (N, (N, T::ASSOC)) }>();
}

fn main() {}
