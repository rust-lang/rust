// run-pass
// ignore-emscripten

#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics, min_const_generics)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct I32<const N: usize>([i32; N]);

extern "platform-intrinsic" {
    fn simd_saturating_add<T>(x: T, y: T) -> T;
    fn simd_saturating_sub<T>(x: T, y: T) -> T;
}

fn main() {
    // unsigned
    {
        const M: u32 = u32::MAX;

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
        const MIN: i32 = i32::MIN;
        const MAX: i32 = i32::MAX;

        let a = I32::<4>([1, 2, 3, 4]);
        let b = I32::<4>([2, 4, 6, 8]);
        let c = I32::<4>([-1, -2, -3, -4]);
        let d = I32::<4>([-2, -4, -6, -8]);

        let max = I32::<4>([MAX, MAX, MAX, MAX]);
        let max1 = I32::<4>([MAX - 1, MAX - 1, MAX - 1, MAX - 1]);
        let min = I32::<4>([MIN, MIN, MIN, MIN]);
        let min1 = I32::<4>([MIN + 1, MIN + 1, MIN + 1, MIN + 1]);

        let z = I32::<4>([0, 0, 0, 0]);

        unsafe {
            assert_eq!(simd_saturating_add(z, z).0, z.0);
            assert_eq!(simd_saturating_add(z, a).0, a.0);
            assert_eq!(simd_saturating_add(b, z).0, b.0);
            assert_eq!(simd_saturating_add(a, a).0, b.0);
            assert_eq!(simd_saturating_add(a, max).0, max.0);
            assert_eq!(simd_saturating_add(max, b).0, max.0);
            assert_eq!(simd_saturating_add(max1, a).0, max.0);
            assert_eq!(simd_saturating_add(min1, z).0, min1.0);
            assert_eq!(simd_saturating_add(min, z).0, min.0);
            assert_eq!(simd_saturating_add(min1, c).0, min.0);
            assert_eq!(simd_saturating_add(min, c).0, min.0);
            assert_eq!(simd_saturating_add(min1, d).0, min.0);
            assert_eq!(simd_saturating_add(min, d).0, min.0);

            assert_eq!(simd_saturating_sub(b, z).0, b.0);
            assert_eq!(simd_saturating_sub(b, a).0, a.0);
            assert_eq!(simd_saturating_sub(a, a).0, z.0);
            assert_eq!(simd_saturating_sub(a, b).0, c.0);
            assert_eq!(simd_saturating_sub(z, max).0, min1.0);
            assert_eq!(simd_saturating_sub(min1, z).0, min1.0);
            assert_eq!(simd_saturating_sub(min1, a).0, min.0);
            assert_eq!(simd_saturating_sub(min1, b).0, min.0);
        }
    }
}
