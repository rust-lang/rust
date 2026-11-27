//@ run-pass

// Test that simd gather instructions on slice of usize don't cause crash
// See issue #89183 - https://github.com/rust-lang/rust/issues/89193

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_gather;

type x4<T> = Simd<T, 4>;

fn main() {
    let x: [usize; 4] = [10, 11, 12, 13];
    let default = x4::from_array([0_usize, 1, 2, 3]);
    let all_set = u8::MAX as i8; // aka -1
    let mask = x4::from_array([all_set, all_set, all_set, all_set]);
    let expected = x4::from_array([10_usize, 11, 12, 13]);

    unsafe {
        let pointer = x.as_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i)));
        let result = simd_gather(default, pointers, mask);
        assert_eq!(result, expected);
    }

    // and again for isize
    let x: [isize; 4] = [10, 11, 12, 13];
    let default = x4::from_array([0_isize, 1, 2, 3]);
    let expected = x4::from_array([10_isize, 11, 12, 13]);

    unsafe {
        let pointer = x.as_ptr();
        let pointers =
            x4::from_array(std::array::from_fn(|i| pointer.add(i)));
        let result = simd_gather(default, pointers, mask);
        assert_eq!(result, expected);
    }
}
