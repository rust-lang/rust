//@run-rustfix
//@aux-build:proc_macros.rs:proc-macro
#![allow(unused)]
#![warn(clippy::manual_slice_size_calculation)]

extern crate proc_macros;

use core::mem::{align_of, size_of};
use proc_macros::external;

fn main() {
    let v_i32 = Vec::<i32>::new();
    let s_i32 = v_i32.as_slice();

    // True positives:
    let _ = s_i32.len() * size_of::<i32>(); // WARNING
    let _ = size_of::<i32>() * s_i32.len(); // WARNING
    let _ = size_of::<i32>() * s_i32.len() * 5; // WARNING

    let len = s_i32.len();
    let size = size_of::<i32>();
    let _ = len * size_of::<i32>(); // WARNING
    let _ = s_i32.len() * size; // WARNING
    let _ = len * size; // WARNING

    let _ = external!(&[1u64][..]).len() * size_of::<u64>();

    // True negatives:
    let _ = size_of::<i32>() + s_i32.len(); // Ok, not a multiplication
    let _ = size_of::<i32>() * s_i32.partition_point(|_| true); // Ok, not len()
    let _ = size_of::<i32>() * v_i32.len(); // Ok, not a slice
    let _ = align_of::<i32>() * s_i32.len(); // Ok, not size_of()
    let _ = size_of::<u32>() * s_i32.len(); // Ok, different types

    let _ = external!($s_i32.len() * size_of::<i32>());
    let _ = external!($s_i32.len()) * size_of::<i32>();

    // False negatives:
    let _ = 5 * size_of::<i32>() * s_i32.len(); // Ok (MISSED OPPORTUNITY)
    let _ = size_of::<i32>() * 5 * s_i32.len(); // Ok (MISSED OPPORTUNITY)
}

const fn _const(s_i32: &[i32]) {
    // True negative:
    let _ = s_i32.len() * size_of::<i32>(); // Ok, can't use size_of_val in const
}
