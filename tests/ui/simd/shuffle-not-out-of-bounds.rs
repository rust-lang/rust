// build-fail
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

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

extern "platform-intrinsic" {
    pub fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    pub fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    pub fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
    pub fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
    pub fn simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U;
    pub fn simd_shuffle64<T, U>(x: T, y: T, idx: [u32; 64]) -> U;
}

// Test vectors by lane size. Since LLVM does not distinguish between a shuffle
// over two f32s and a shuffle over two u64s, or any other such combination,
// it is not necessary to test every possible vector, only lane counts.
macro_rules! test_shuffle_lanes {
    ($n:literal, $x:ident, $y:ident) => {
        unsafe {
                let shuffle: $x = {
                    const ARR: [u32; $n] = {
                        let mut arr = [0; $n];
                        arr[0] = $n * 2;
                        arr
                    };
                    let mut n: u8 = $n;
                    let vals = [0; $n].map(|_| { n = n - 1; n });
                    let vec1 = $x(vals);
                    let vec2 = $x(vals);
                    $y(vec1, vec2, ARR)
                };
        }
    }
}
//~^^^^^ ERROR: invalid monomorphization of `simd_shuffle2` intrinsic
//~| ERROR: invalid monomorphization of `simd_shuffle4` intrinsic
//~| ERROR: invalid monomorphization of `simd_shuffle8` intrinsic
//~| ERROR: invalid monomorphization of `simd_shuffle16` intrinsic
//~| ERROR: invalid monomorphization of `simd_shuffle32` intrinsic
//~| ERROR: invalid monomorphization of `simd_shuffle64` intrinsic
// Because the test is mostly embedded in a macro, all the errors have the same origin point.
// And unfortunately, standard comments, as in the UI test harness, disappear in macros!

fn main() {
    test_shuffle_lanes!(2, u8x2, simd_shuffle2);
    test_shuffle_lanes!(4, u8x4, simd_shuffle4);
    test_shuffle_lanes!(8, u8x8, simd_shuffle8);
    test_shuffle_lanes!(16, u8x16, simd_shuffle16);
    test_shuffle_lanes!(32, u8x32, simd_shuffle32);
    test_shuffle_lanes!(64, u8x64, simd_shuffle64);

    extern "platform-intrinsic" {
        fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
    }
    let v = u8x2([0, 0]);
    const I: [u32; 2] = [4, 4];
    unsafe {
        let _: u8x2 = simd_shuffle(v, v, I);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic
    }
}
