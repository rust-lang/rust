//@ build-fail
//@ ignore-backends: gcc
#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

// Test for #73542 to verify out-of-bounds shuffle vectors do not compile.

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x2([u8; 2]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x4([u8; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x8([u8; 8]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x16([u8; 16]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x32([u8; 32]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x64([u8; 64]);

use std::intrinsics::simd::*;

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

// Test vectors by lane size. Since LLVM does not distinguish between a shuffle
// over two f32s and a shuffle over two u64s, or any other such combination,
// it is not necessary to test every possible vector, only lane counts.
macro_rules! test_shuffle_lanes {
    ($n:literal, $x:ident, $y:ident) => {
        unsafe {
                let shuffle: $x = {
                    const IDX: SimdShuffleIdx<$n> = SimdShuffleIdx({
                        let mut arr = [0; $n];
                        arr[0] = $n * 2;
                        arr
                    });
                    let mut n: u8 = $n;
                    let vals = [0; $n].map(|_| { n = n - 1; n });
                    let vec1 = $x(vals);
                    let vec2 = $x(vals);
                    $y(vec1, vec2, IDX)
                };
        }
    }
}
//~^^^^^ ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
//~| ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
//~| ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
//~| ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
//~| ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
//~| ERROR: invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
// Because the test is mostly embedded in a macro, all the errors have the same origin point.
// And unfortunately, standard comments, as in the UI test harness, disappear in macros!

fn main() {
    test_shuffle_lanes!(2, u8x2, simd_shuffle);
    test_shuffle_lanes!(4, u8x4, simd_shuffle);
    test_shuffle_lanes!(8, u8x8, simd_shuffle);
    test_shuffle_lanes!(16, u8x16, simd_shuffle);
    test_shuffle_lanes!(32, u8x32, simd_shuffle);
    test_shuffle_lanes!(64, u8x64, simd_shuffle);

    let v = u8x2([0, 0]);
    const I: SimdShuffleIdx<2> = SimdShuffleIdx([4, 4]);
    unsafe {
        let _: u8x2 = simd_shuffle(v, v, I);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic: SIMD index #0 is out of bounds
    }

    // also check insert/extract
    unsafe {
        simd_insert(v, 2, 0u8); //~ ERROR invalid monomorphization of `simd_insert` intrinsic: SIMD index #1 is out of bounds
        let _val: u8 = simd_extract(v, 2); //~ ERROR invalid monomorphization of `simd_extract` intrinsic: SIMD index #1 is out of bounds
    }
}
