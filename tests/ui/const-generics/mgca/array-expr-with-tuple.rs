//@ run-pass
#![feature(
    gca_min_const_items,
    gca_macroless_args,
    adt_const_params,
    unsized_const_params
)]
#![expect(incomplete_features)]
#![allow(dead_code)]

fn takes_tuple<const T: ([u32; 2], u32, [u32; 2])>() {}

fn generic_caller<const N: u32, const M: u32>() {
    takes_tuple::<{ ([N, M], 5, [M, N]) }>();
    takes_tuple::<{ ([1, 2], 3, [4, 5]) }>();
}

fn main() {}
