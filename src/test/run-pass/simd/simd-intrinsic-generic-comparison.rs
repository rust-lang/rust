// run-pass
// ignore-emscripten FIXME(#45351) hits an LLVM assert

#![feature(repr_simd, platform_intrinsics, concat_idents)]
#![allow(non_camel_case_types)]

use std::f32::NAN;

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
struct u32x4(pub u32, pub u32, pub u32, pub u32);
#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_eq<T, U>(x: T, y: T) -> U;
    fn simd_ne<T, U>(x: T, y: T) -> U;
    fn simd_lt<T, U>(x: T, y: T) -> U;
    fn simd_le<T, U>(x: T, y: T) -> U;
    fn simd_gt<T, U>(x: T, y: T) -> U;
    fn simd_ge<T, U>(x: T, y: T) -> U;
}

macro_rules! cmp {
    ($method: ident($lhs: expr, $rhs: expr)) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let e: u32x4 = concat_idents!(simd_, $method)($lhs, $rhs);
        // assume the scalar version is correct/the behaviour we want.
        assert!((e.0 != 0) == lhs.0 .$method(&rhs.0));
        assert!((e.1 != 0) == lhs.1 .$method(&rhs.1));
        assert!((e.2 != 0) == lhs.2 .$method(&rhs.2));
        assert!((e.3 != 0) == lhs.3 .$method(&rhs.3));
    }}
}
macro_rules! tests {
    ($($lhs: ident, $rhs: ident;)*) => {{
        $(
            (|| {
                cmp!(eq($lhs, $rhs));
                cmp!(ne($lhs, $rhs));

                // test both directions
                cmp!(lt($lhs, $rhs));
                cmp!(lt($rhs, $lhs));

                cmp!(le($lhs, $rhs));
                cmp!(le($rhs, $lhs));

                cmp!(gt($lhs, $rhs));
                cmp!(gt($rhs, $lhs));

                cmp!(ge($lhs, $rhs));
                cmp!(ge($rhs, $lhs));
            })();
            )*
    }}
}
fn main() {
    // 13 vs. -100 tests that we get signed vs. unsigned comparisons
    // correct (i32: 13 > -100, u32: 13 < -100).    let i1 = i32x4(10, -11, 12, 13);
    let i1 = i32x4(10, -11, 12, 13);
    let i2 = i32x4(5, -5, 20, -100);
    let i3 = i32x4(10, -11, 20, -100);

    let u1 = u32x4(10, !11+1, 12, 13);
    let u2 = u32x4(5, !5+1, 20, !100+1);
    let u3 = u32x4(10, !11+1, 20, !100+1);

    let f1 = f32x4(10.0, -11.0, 12.0, 13.0);
    let f2 = f32x4(5.0, -5.0, 20.0, -100.0);
    let f3 = f32x4(10.0, -11.0, 20.0, -100.0);

    unsafe {
        tests! {
            i1, i1;
            u1, u1;
            f1, f1;

            i1, i2;
            u1, u2;
            f1, f2;

            i1, i3;
            u1, u3;
            f1, f3;
        }
    }

    // NAN comparisons are special:
    // -11 (*)    13
    // -5        -100 (*)
    let f4 = f32x4(NAN, f1.1, NAN, f2.3);

    unsafe {
        tests! {
            f1, f4;
            f2, f4;
            f4, f4;
        }
    }
}
