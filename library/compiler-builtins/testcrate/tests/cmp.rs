#![allow(unused_macros)]

use testcrate::*;

macro_rules! cmp {
    ($x:ident, $y:ident, $($unordered_val:expr, $fn:ident);*;) => {
        $(
            let cmp0 = if $x.is_nan() || $y.is_nan() {
                $unordered_val
            } else if $x < $y {
                -1
            } else if $x == $y {
                0
            } else {
                1
            };
            let cmp1 = $fn($x, $y);
            if cmp0 != cmp1 {
                panic!("{}({}, {}): std: {}, builtins: {}", stringify!($fn_builtins), $x, $y, cmp0, cmp1);
            }
        )*
    };
}

#[test]
fn float_comparisons() {
    use compiler_builtins::float::cmp::{
        __eqdf2, __eqsf2, __gedf2, __gesf2, __gtdf2, __gtsf2, __ledf2, __lesf2, __ltdf2, __ltsf2,
        __nedf2, __nesf2, __unorddf2, __unordsf2,
    };

    fuzz_float_2(N, |x: f32, y: f32| {
        assert_eq!(__unordsf2(x, y) != 0, x.is_nan() || y.is_nan());
        cmp!(x, y,
            1, __ltsf2;
            1, __lesf2;
            1, __eqsf2;
            -1, __gesf2;
            -1, __gtsf2;
            1, __nesf2;
        );
    });
    fuzz_float_2(N, |x: f64, y: f64| {
        assert_eq!(__unorddf2(x, y) != 0, x.is_nan() || y.is_nan());
        cmp!(x, y,
            1, __ltdf2;
            1, __ledf2;
            1, __eqdf2;
            -1, __gedf2;
            -1, __gtdf2;
            1, __nedf2;
        );
    });
}

macro_rules! cmp2 {
    ($x:ident, $y:ident, $($unordered_val:expr, $fn_std:expr, $fn_builtins:ident);*;) => {
        $(
            let cmp0: i32 = if $x.is_nan() || $y.is_nan() {
                $unordered_val
            } else {
                $fn_std as i32
            };
            let cmp1: i32 = $fn_builtins($x, $y);
            if cmp0 != cmp1 {
                panic!("{}({}, {}): std: {}, builtins: {}", stringify!($fn_builtins), $x, $y, cmp0, cmp1);
            }
        )*
    };
}

#[cfg(target_arch = "arm")]
#[test]
fn float_comparisons_arm() {
    use compiler_builtins::float::cmp::{
        __aeabi_dcmpeq, __aeabi_dcmpge, __aeabi_dcmpgt, __aeabi_dcmple, __aeabi_dcmplt,
        __aeabi_fcmpeq, __aeabi_fcmpge, __aeabi_fcmpgt, __aeabi_fcmple, __aeabi_fcmplt, __eqdf2vfp,
        __eqsf2vfp, __gedf2vfp, __gesf2vfp, __gtdf2vfp, __gtsf2vfp, __ledf2vfp, __lesf2vfp,
        __ltdf2vfp, __ltsf2vfp, __nedf2vfp, __nesf2vfp,
    };

    fuzz_float_2(N, |x: f32, y: f32| {
        cmp2!(x, y,
            0, x < y, __aeabi_fcmplt;
            0, x <= y, __aeabi_fcmple;
            0, x == y, __aeabi_fcmpeq;
            0, x >= y, __aeabi_fcmpge;
            0, x > y, __aeabi_fcmpgt;
            0, x < y, __ltsf2vfp;
            0, x <= y, __lesf2vfp;
            0, x == y, __eqsf2vfp;
            0, x >= y, __gesf2vfp;
            0, x > y, __gtsf2vfp;
            1, x != y, __nesf2vfp;
        );
    });
    fuzz_float_2(N, |x: f64, y: f64| {
        cmp2!(x, y,
            0, x < y, __aeabi_dcmplt;
            0, x <= y, __aeabi_dcmple;
            0, x == y, __aeabi_dcmpeq;
            0, x >= y, __aeabi_dcmpge;
            0, x > y, __aeabi_dcmpgt;
            0, x < y, __ltdf2vfp;
            0, x <= y, __ledf2vfp;
            0, x == y, __eqdf2vfp;
            0, x >= y, __gedf2vfp;
            0, x > y, __gtdf2vfp;
            1, x != y, __nedf2vfp;
        );
    });
}
