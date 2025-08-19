//@ run-pass

#![feature(repr_simd, core_intrinsics, macro_metavar_expr_concat)]
#![allow(non_camel_case_types)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_eq, simd_ge, simd_gt, simd_le, simd_lt, simd_ne};

macro_rules! cmp {
    ($method: ident($lhs: expr, $rhs: expr)) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let e: u32x4 = ${concat(simd_, $method)}($lhs, $rhs);
        // assume the scalar version is correct/the behaviour we want.
        let (lhs, rhs, e) = (lhs.as_array(), rhs.as_array(), e.as_array());
        assert!((e[0] != 0) == lhs[0].$method(&rhs[0]));
        assert!((e[1] != 0) == lhs[1].$method(&rhs[1]));
        assert!((e[2] != 0) == lhs[2].$method(&rhs[2]));
        assert!((e[3] != 0) == lhs[3].$method(&rhs[3]));
    }};
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
    let i1 = i32x4::from_array([10, -11, 12, 13]);
    let i2 = i32x4::from_array([5, -5, 20, -100]);
    let i3 = i32x4::from_array([10, -11, 20, -100]);

    let u1 = u32x4::from_array([10, !11 + 1, 12, 13]);
    let u2 = u32x4::from_array([5, !5 + 1, 20, !100 + 1]);
    let u3 = u32x4::from_array([10, !11 + 1, 20, !100 + 1]);

    let f1 = f32x4::from_array([10.0, -11.0, 12.0, 13.0]);
    let f2 = f32x4::from_array([5.0, -5.0, 20.0, -100.0]);
    let f3 = f32x4::from_array([10.0, -11.0, 20.0, -100.0]);

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
    let f4 = f32x4::from_array([f32::NAN, f1[1], f32::NAN, f2[3]]);

    unsafe {
        tests! {
            f1, f4;
            f2, f4;
            f4, f4;
        }
    }
}
