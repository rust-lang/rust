// run-pass
#![allow(non_camel_case_types)]

// ignore-emscripten FIXME(#45351) hits an LLVM assert

#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct U32<const N: usize>([u32; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

macro_rules! all_eq {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        assert!(a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3);
    }}
}

macro_rules! all_eq_ {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        assert!(a.0 == b.0);
    }}
}


extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;
    fn simd_sub<T>(x: T, y: T) -> T;
    fn simd_mul<T>(x: T, y: T) -> T;
    fn simd_div<T>(x: T, y: T) -> T;
    fn simd_rem<T>(x: T, y: T) -> T;
    fn simd_shl<T>(x: T, y: T) -> T;
    fn simd_shr<T>(x: T, y: T) -> T;
    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
    fn simd_xor<T>(x: T, y: T) -> T;
}

fn main() {
    let x1 = i32x4(1, 2, 3, 4);
    let y1 = U32::<4>([1, 2, 3, 4]);
    let z1 = f32x4(1.0, 2.0, 3.0, 4.0);
    let x2 = i32x4(2, 3, 4, 5);
    let y2 = U32::<4>([2, 3, 4, 5]);
    let z2 = f32x4(2.0, 3.0, 4.0, 5.0);

    unsafe {
        all_eq!(simd_add(x1, x2), i32x4(3, 5, 7, 9));
        all_eq!(simd_add(x2, x1), i32x4(3, 5, 7, 9));
        all_eq_!(simd_add(y1, y2), U32::<4>([3, 5, 7, 9]));
        all_eq_!(simd_add(y2, y1), U32::<4>([3, 5, 7, 9]));
        all_eq!(simd_add(z1, z2), f32x4(3.0, 5.0, 7.0, 9.0));
        all_eq!(simd_add(z2, z1), f32x4(3.0, 5.0, 7.0, 9.0));

        all_eq!(simd_mul(x1, x2), i32x4(2, 6, 12, 20));
        all_eq!(simd_mul(x2, x1), i32x4(2, 6, 12, 20));
        all_eq_!(simd_mul(y1, y2), U32::<4>([2, 6, 12, 20]));
        all_eq_!(simd_mul(y2, y1), U32::<4>([2, 6, 12, 20]));
        all_eq!(simd_mul(z1, z2), f32x4(2.0, 6.0, 12.0, 20.0));
        all_eq!(simd_mul(z2, z1), f32x4(2.0, 6.0, 12.0, 20.0));

        all_eq!(simd_sub(x2, x1), i32x4(1, 1, 1, 1));
        all_eq!(simd_sub(x1, x2), i32x4(-1, -1, -1, -1));
        all_eq_!(simd_sub(y2, y1), U32::<4>([1, 1, 1, 1]));
        all_eq_!(simd_sub(y1, y2), U32::<4>([!0, !0, !0, !0]));
        all_eq!(simd_sub(z2, z1), f32x4(1.0, 1.0, 1.0, 1.0));
        all_eq!(simd_sub(z1, z2), f32x4(-1.0, -1.0, -1.0, -1.0));

        all_eq!(simd_div(x1, x1), i32x4(1, 1, 1, 1));
        all_eq!(simd_div(i32x4(2, 4, 6, 8), i32x4(2, 2, 2, 2)), x1);
        all_eq_!(simd_div(y1, y1), U32::<4>([1, 1, 1, 1]));
        all_eq_!(simd_div(U32::<4>([2, 4, 6, 8]), U32::<4>([2, 2, 2, 2])), y1);
        all_eq!(simd_div(z1, z1), f32x4(1.0, 1.0, 1.0, 1.0));
        all_eq!(simd_div(z1, z2), f32x4(1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0));
        all_eq!(simd_div(z2, z1), f32x4(2.0/1.0, 3.0/2.0, 4.0/3.0, 5.0/4.0));

        all_eq!(simd_rem(x1, x1), i32x4(0, 0, 0, 0));
        all_eq!(simd_rem(x2, x1), i32x4(0, 1, 1, 1));
        all_eq_!(simd_rem(y1, y1), U32::<4>([0, 0, 0, 0]));
        all_eq_!(simd_rem(y2, y1), U32::<4>([0, 1, 1, 1]));
        all_eq!(simd_rem(z1, z1), f32x4(0.0, 0.0, 0.0, 0.0));
        all_eq!(simd_rem(z1, z2), z1);
        all_eq!(simd_rem(z2, z1), f32x4(0.0, 1.0, 1.0, 1.0));

        all_eq!(simd_shl(x1, x2), i32x4(1 << 2, 2 << 3, 3 << 4, 4 << 5));
        all_eq!(simd_shl(x2, x1), i32x4(2 << 1, 3 << 2, 4 << 3, 5 << 4));
        all_eq_!(simd_shl(y1, y2), U32::<4>([1 << 2, 2 << 3, 3 << 4, 4 << 5]));
        all_eq_!(simd_shl(y2, y1), U32::<4>([2 << 1, 3 << 2, 4 << 3, 5 << 4]));

        // test right-shift by assuming left-shift is correct
        all_eq!(simd_shr(simd_shl(x1, x2), x2), x1);
        all_eq!(simd_shr(simd_shl(x2, x1), x1), x2);
        all_eq_!(simd_shr(simd_shl(y1, y2), y2), y1);
        all_eq_!(simd_shr(simd_shl(y2, y1), y1), y2);

        // ensure we get logical vs. arithmetic shifts correct
        let (a, b, c, d) = (-12, -123, -1234, -12345);
        all_eq!(simd_shr(i32x4(a, b, c, d), x1), i32x4(a >> 1, b >> 2, c >> 3, d >> 4));
        all_eq_!(simd_shr(U32::<4>([a as u32, b as u32, c as u32, d as u32]), y1),
                U32::<4>([(a as u32) >> 1, (b as u32) >> 2, (c as u32) >> 3, (d as u32) >> 4]));

        all_eq!(simd_and(x1, x2), i32x4(0, 2, 0, 4));
        all_eq!(simd_and(x2, x1), i32x4(0, 2, 0, 4));
        all_eq_!(simd_and(y1, y2), U32::<4>([0, 2, 0, 4]));
        all_eq_!(simd_and(y2, y1), U32::<4>([0, 2, 0, 4]));

        all_eq!(simd_or(x1, x2), i32x4(3, 3, 7, 5));
        all_eq!(simd_or(x2, x1), i32x4(3, 3, 7, 5));
        all_eq_!(simd_or(y1, y2), U32::<4>([3, 3, 7, 5]));
        all_eq_!(simd_or(y2, y1), U32::<4>([3, 3, 7, 5]));

        all_eq!(simd_xor(x1, x2), i32x4(3, 1, 7, 1));
        all_eq!(simd_xor(x2, x1), i32x4(3, 1, 7, 1));
        all_eq_!(simd_xor(y1, y2), U32::<4>([3, 1, 7, 1]));
        all_eq_!(simd_xor(y2, y1), U32::<4>([3, 1, 7, 1]));

    }
}
