// run-pass
// ignore-emscripten
// min-llvm-version 8.0

#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct i32x4(pub i32, pub i32, pub i32, pub i32);

extern "platform-intrinsic" {
    fn simd_saturating_add<T>(x: T, y: T) -> T;
    fn simd_saturating_sub<T>(x: T, y: T) -> T;
}

fn main() {
    // unsigned
    {
        const M: u32 = u32::max_value();

        let a = u32x4(1, 2, 3, 4);
        let b = u32x4(2, 4, 6, 8);
        let m = u32x4(M, M, M, M);
        let m1 = u32x4(M - 1, M - 1, M - 1, M - 1);
        let z = u32x4(0, 0, 0, 0);

        unsafe {
            assert_eq!(simd_saturating_add(z, z), z);
            assert_eq!(simd_saturating_add(z, a), a);
            assert_eq!(simd_saturating_add(b, z), b);
            assert_eq!(simd_saturating_add(a, a), b);
            assert_eq!(simd_saturating_add(a, m), m);
            assert_eq!(simd_saturating_add(m, b), m);
            assert_eq!(simd_saturating_add(m1, a), m);

            assert_eq!(simd_saturating_sub(b, z), b);
            assert_eq!(simd_saturating_sub(b, a), a);
            assert_eq!(simd_saturating_sub(a, a), z);
            assert_eq!(simd_saturating_sub(a, b), z);
            assert_eq!(simd_saturating_sub(a, m1), z);
            assert_eq!(simd_saturating_sub(b, m1), z);
        }
    }

    // signed
    {
        const MIN: i32 = i32::min_value();
        const MAX: i32 = i32::max_value();

        let a = i32x4(1, 2, 3, 4);
        let b = i32x4(2, 4, 6, 8);
        let c = i32x4(-1, -2, -3, -4);
        let d = i32x4(-2, -4, -6, -8);

        let max = i32x4(MAX, MAX, MAX, MAX);
        let max1 = i32x4(MAX - 1, MAX - 1, MAX - 1, MAX - 1);
        let min = i32x4(MIN, MIN, MIN, MIN);
        let min1 = i32x4(MIN + 1, MIN + 1, MIN + 1, MIN + 1);

        let z = i32x4(0, 0, 0, 0);

        unsafe {
            assert_eq!(simd_saturating_add(z, z), z);
            assert_eq!(simd_saturating_add(z, a), a);
            assert_eq!(simd_saturating_add(b, z), b);
            assert_eq!(simd_saturating_add(a, a), b);
            assert_eq!(simd_saturating_add(a, max), max);
            assert_eq!(simd_saturating_add(max, b), max);
            assert_eq!(simd_saturating_add(max1, a), max);
            assert_eq!(simd_saturating_add(min1, z), min1);
            assert_eq!(simd_saturating_add(min, z), min);
            assert_eq!(simd_saturating_add(min1, c), min);
            assert_eq!(simd_saturating_add(min, c), min);
            assert_eq!(simd_saturating_add(min1, d), min);
            assert_eq!(simd_saturating_add(min, d), min);

            assert_eq!(simd_saturating_sub(b, z), b);
            assert_eq!(simd_saturating_sub(b, a), a);
            assert_eq!(simd_saturating_sub(a, a), z);
            assert_eq!(simd_saturating_sub(a, b), c);
            assert_eq!(simd_saturating_sub(z, max), min1);
            assert_eq!(simd_saturating_sub(min1, z), min1);
            assert_eq!(simd_saturating_sub(min1, a), min);
            assert_eq!(simd_saturating_sub(min1, b), min);
        }
    }
}
