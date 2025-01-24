//@ run-pass
#![allow(non_camel_case_types)]
//@ ignore-emscripten FIXME(#45351) hits an LLVM assert
#![feature(repr_simd, intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x4(pub [i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct U32<const N: usize>([u32; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(pub [f32; 4]);

macro_rules! all_eq {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        assert!(a.0 == b.0);
    }};
}

extern "rust-intrinsic" {
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

    fn simd_neg<T>(x: T) -> T;
    fn simd_bswap<T>(x: T) -> T;
    fn simd_bitreverse<T>(x: T) -> T;
    fn simd_ctlz<T>(x: T) -> T;
    fn simd_ctpop<T>(x: T) -> T;
    fn simd_cttz<T>(x: T) -> T;
}

fn main() {
    let x1 = i32x4([1, 2, 3, 4]);
    let y1 = U32::<4>([1, 2, 3, 4]);
    let z1 = f32x4([1.0, 2.0, 3.0, 4.0]);
    let x2 = i32x4([2, 3, 4, 5]);
    let y2 = U32::<4>([2, 3, 4, 5]);
    let z2 = f32x4([2.0, 3.0, 4.0, 5.0]);
    let x3 = i32x4([0, i32::MAX, i32::MIN, -1_i32]);
    let y3 = U32::<4>([0, i32::MAX as _, i32::MIN as _, -1_i32 as _]);

    unsafe {
        all_eq!(simd_add(x1, x2), i32x4([3, 5, 7, 9]));
        all_eq!(simd_add(x2, x1), i32x4([3, 5, 7, 9]));
        all_eq!(simd_add(y1, y2), U32::<4>([3, 5, 7, 9]));
        all_eq!(simd_add(y2, y1), U32::<4>([3, 5, 7, 9]));
        all_eq!(simd_add(z1, z2), f32x4([3.0, 5.0, 7.0, 9.0]));
        all_eq!(simd_add(z2, z1), f32x4([3.0, 5.0, 7.0, 9.0]));

        all_eq!(simd_mul(x1, x2), i32x4([2, 6, 12, 20]));
        all_eq!(simd_mul(x2, x1), i32x4([2, 6, 12, 20]));
        all_eq!(simd_mul(y1, y2), U32::<4>([2, 6, 12, 20]));
        all_eq!(simd_mul(y2, y1), U32::<4>([2, 6, 12, 20]));
        all_eq!(simd_mul(z1, z2), f32x4([2.0, 6.0, 12.0, 20.0]));
        all_eq!(simd_mul(z2, z1), f32x4([2.0, 6.0, 12.0, 20.0]));

        all_eq!(simd_sub(x2, x1), i32x4([1, 1, 1, 1]));
        all_eq!(simd_sub(x1, x2), i32x4([-1, -1, -1, -1]));
        all_eq!(simd_sub(y2, y1), U32::<4>([1, 1, 1, 1]));
        all_eq!(simd_sub(y1, y2), U32::<4>([!0, !0, !0, !0]));
        all_eq!(simd_sub(z2, z1), f32x4([1.0, 1.0, 1.0, 1.0]));
        all_eq!(simd_sub(z1, z2), f32x4([-1.0, -1.0, -1.0, -1.0]));

        all_eq!(simd_div(x1, x1), i32x4([1, 1, 1, 1]));
        all_eq!(simd_div(i32x4([2, 4, 6, 8]), i32x4([2, 2, 2, 2])), x1);
        all_eq!(simd_div(y1, y1), U32::<4>([1, 1, 1, 1]));
        all_eq!(simd_div(U32::<4>([2, 4, 6, 8]), U32::<4>([2, 2, 2, 2])), y1);
        all_eq!(simd_div(z1, z1), f32x4([1.0, 1.0, 1.0, 1.0]));
        all_eq!(simd_div(z1, z2), f32x4([1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0, 4.0 / 5.0]));
        all_eq!(simd_div(z2, z1), f32x4([2.0 / 1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0]));

        all_eq!(simd_rem(x1, x1), i32x4([0, 0, 0, 0]));
        all_eq!(simd_rem(x2, x1), i32x4([0, 1, 1, 1]));
        all_eq!(simd_rem(y1, y1), U32::<4>([0, 0, 0, 0]));
        all_eq!(simd_rem(y2, y1), U32::<4>([0, 1, 1, 1]));
        all_eq!(simd_rem(z1, z1), f32x4([0.0, 0.0, 0.0, 0.0]));
        all_eq!(simd_rem(z1, z2), z1);
        all_eq!(simd_rem(z2, z1), f32x4([0.0, 1.0, 1.0, 1.0]));

        all_eq!(simd_shl(x1, x2), i32x4([1 << 2, 2 << 3, 3 << 4, 4 << 5]));
        all_eq!(simd_shl(x2, x1), i32x4([2 << 1, 3 << 2, 4 << 3, 5 << 4]));
        all_eq!(simd_shl(y1, y2), U32::<4>([1 << 2, 2 << 3, 3 << 4, 4 << 5]));
        all_eq!(simd_shl(y2, y1), U32::<4>([2 << 1, 3 << 2, 4 << 3, 5 << 4]));

        // test right-shift by assuming left-shift is correct
        all_eq!(simd_shr(simd_shl(x1, x2), x2), x1);
        all_eq!(simd_shr(simd_shl(x2, x1), x1), x2);
        all_eq!(simd_shr(simd_shl(y1, y2), y2), y1);
        all_eq!(simd_shr(simd_shl(y2, y1), y1), y2);

        // ensure we get logical vs. arithmetic shifts correct
        let (a, b, c, d) = (-12, -123, -1234, -12345);
        all_eq!(simd_shr(i32x4([a, b, c, d]), x1), i32x4([a >> 1, b >> 2, c >> 3, d >> 4]));
        all_eq!(
            simd_shr(U32::<4>([a as u32, b as u32, c as u32, d as u32]), y1),
            U32::<4>([(a as u32) >> 1, (b as u32) >> 2, (c as u32) >> 3, (d as u32) >> 4])
        );

        all_eq!(simd_and(x1, x2), i32x4([0, 2, 0, 4]));
        all_eq!(simd_and(x2, x1), i32x4([0, 2, 0, 4]));
        all_eq!(simd_and(y1, y2), U32::<4>([0, 2, 0, 4]));
        all_eq!(simd_and(y2, y1), U32::<4>([0, 2, 0, 4]));

        all_eq!(simd_or(x1, x2), i32x4([3, 3, 7, 5]));
        all_eq!(simd_or(x2, x1), i32x4([3, 3, 7, 5]));
        all_eq!(simd_or(y1, y2), U32::<4>([3, 3, 7, 5]));
        all_eq!(simd_or(y2, y1), U32::<4>([3, 3, 7, 5]));

        all_eq!(simd_xor(x1, x2), i32x4([3, 1, 7, 1]));
        all_eq!(simd_xor(x2, x1), i32x4([3, 1, 7, 1]));
        all_eq!(simd_xor(y1, y2), U32::<4>([3, 1, 7, 1]));
        all_eq!(simd_xor(y2, y1), U32::<4>([3, 1, 7, 1]));

        all_eq!(simd_neg(x1), i32x4([-1, -2, -3, -4]));
        all_eq!(simd_neg(x2), i32x4([-2, -3, -4, -5]));
        all_eq!(simd_neg(z1), f32x4([-1.0, -2.0, -3.0, -4.0]));
        all_eq!(simd_neg(z2), f32x4([-2.0, -3.0, -4.0, -5.0]));

        all_eq!(simd_bswap(x1), i32x4([0x01000000, 0x02000000, 0x03000000, 0x04000000]));
        all_eq!(simd_bswap(y1), U32::<4>([0x01000000, 0x02000000, 0x03000000, 0x04000000]));

        all_eq!(
            simd_bitreverse(x1),
            i32x4([0x80000000u32 as i32, 0x40000000, 0xc0000000u32 as i32, 0x20000000])
        );
        all_eq!(simd_bitreverse(y1), U32::<4>([0x80000000, 0x40000000, 0xc0000000, 0x20000000]));

        all_eq!(simd_ctlz(x1), i32x4([31, 30, 30, 29]));
        all_eq!(simd_ctlz(y1), U32::<4>([31, 30, 30, 29]));

        all_eq!(simd_ctpop(x1), i32x4([1, 1, 2, 1]));
        all_eq!(simd_ctpop(y1), U32::<4>([1, 1, 2, 1]));
        all_eq!(simd_ctpop(x2), i32x4([1, 2, 1, 2]));
        all_eq!(simd_ctpop(y2), U32::<4>([1, 2, 1, 2]));
        all_eq!(simd_ctpop(x3), i32x4([0, 31, 1, 32]));
        all_eq!(simd_ctpop(y3), U32::<4>([0, 31, 1, 32]));

        all_eq!(simd_cttz(x1), i32x4([0, 1, 0, 2]));
        all_eq!(simd_cttz(y1), U32::<4>([0, 1, 0, 2]));
    }
}
