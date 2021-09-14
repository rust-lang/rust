// build-fail
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

// Test for #73542 to verify out-of-bounds shuffle vectors do not compile.

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x2(u8, u8);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x4(u8, u8, u8, u8);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x8(u8, u8, u8, u8, u8, u8, u8, u8);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x16(
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x32(
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x64(
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
    u8,
);

// Test vectors by lane size. Since LLVM does not distinguish between a shuffle
// over two f32s and a shuffle over two u64s, or any other such combination,
// it is not necessary to test every possible vector, only lane counts.
macro_rules! test_shuffle_lanes {
    ($n:literal, $x:ident, $y:ident, $t:tt) => {
        unsafe {
                let shuffle: $x = {
                    const ARR: [u32; $n] = {
                        let mut arr = [0; $n];
                        arr[0] = $n * 2;
                        arr
                    };
                    extern "platform-intrinsic" {
                        pub fn $y<T, U>(x: T, y: T, idx: [u32; $n]) -> U;
                    }
                    let vec1 = $x$t;
                    let vec2 = $x$t;
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
    test_shuffle_lanes!(2, u8x2, simd_shuffle2, (2, 1));
    test_shuffle_lanes!(4, u8x4, simd_shuffle4, (4, 3, 2, 1));
    test_shuffle_lanes!(8, u8x8, simd_shuffle8, (8, 7, 6, 5, 4, 3, 2, 1));
    test_shuffle_lanes!(16, u8x16, simd_shuffle16,
        (16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    test_shuffle_lanes!(32, u8x32, simd_shuffle32,
        (32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
         15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    test_shuffle_lanes!(64, u8x64, simd_shuffle64,
        (64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
         48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
         32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
         16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));

    extern "platform-intrinsic" {
        fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
    }
    let v = u8x2(0, 0);
    const I: [u32; 2] = [4, 4];
    unsafe {
        let _: u8x2 = simd_shuffle(v, v, I);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic
    }
}
