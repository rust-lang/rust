//@ run-pass
#![feature(
    adt_const_params,
    gca_adts,
    gca_macroless_args,
    gca_min_const_items,
    unsized_const_params
)]
#![allow(dead_code)]

fn takes_tuple<const T: ([u32; 2], u32, [u32; 2])>() {}

fn generic_caller<const N: u32, const M: u32>() {
    takes_tuple::<{ ([N, M], 5, [M, N]) }>();
    takes_tuple::<{ ([1, 2], 3, [4, 5]) }>();
}

fn main() {}
