//@aux-build:proc_macros.rs
#![allow(unused)]
#![warn(clippy::manual_slice_size_calculation)]

extern crate proc_macros;

use core::mem::{align_of, size_of};
use proc_macros::external;

fn main() {
    let v_i32 = Vec::<i32>::new();
    let s_i32 = v_i32.as_slice();
    let s_i32_ref = &s_i32;
    let s_i32_ref_ref = &s_i32_ref;

    // True positives:
    let _ = s_i32.len() * size_of::<i32>(); // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = size_of::<i32>() * s_i32.len(); // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = size_of::<i32>() * s_i32.len() * 5; // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = size_of::<i32>() * s_i32_ref.len(); // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = size_of::<i32>() * s_i32_ref_ref.len(); // WARNING
    //
    //~^^ manual_slice_size_calculation

    let len = s_i32.len();
    let size = size_of::<i32>();
    let _ = len * size_of::<i32>(); // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = s_i32.len() * size; // WARNING
    //
    //~^^ manual_slice_size_calculation
    let _ = len * size; // WARNING
    //
    //~^^ manual_slice_size_calculation

    let _ = external!(&[1u64][..]).len() * size_of::<u64>();
    //~^ manual_slice_size_calculation

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

#[clippy::msrv = "1.85"]
const fn const_ok(s_i32: &[i32]) {
    let _ = s_i32.len() * size_of::<i32>();
    //~^ manual_slice_size_calculation
}

#[clippy::msrv = "1.84"]
const fn const_before_msrv(s_i32: &[i32]) {
    let _ = s_i32.len() * size_of::<i32>();
}

fn issue_14802() {
    struct IcedSlice {
        dst: [u8],
    }

    impl IcedSlice {
        fn get_len(&self) -> usize {
            self.dst.len() * size_of::<u8>()
            //~^ manual_slice_size_calculation
        }
    }
}
