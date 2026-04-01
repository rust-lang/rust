//@ run-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]
#![allow(dead_code)]

fn takes_array_u32<const A: [u32; 3]>() {}
fn takes_array_bool<const A: [bool; 2]>() {}
fn takes_nested_array<const A: [[u32; 2]; 2]>() {}
fn takes_empty_array<const A: [u32; 0]>() {}

fn generic_caller<const X: u32, const Y: u32>() {
    takes_array_u32::<{ [X, Y, X] }>();
    takes_array_u32::<{ [X, Y, const { 1 }] }>();
    takes_array_u32::<{ [X, Y, const { 1 + 1 }] }>();
    takes_array_u32::<{ [2_002, 2u32, 1_u32] }>();

    takes_array_bool::<{ [true, false] }>();

    takes_nested_array::<{ [[X, Y], [3, 4]] }>();
    takes_nested_array::<{ [[1u32, 2_u32], [const { 3 }, 4]] }>();

    takes_empty_array::<{ [] }>();
}

fn main() {}
