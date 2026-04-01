#![allow(unused_macros)]
#![allow(unreachable_code)]
#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::*;

mod float_comparisons {
    use super::*;

    macro_rules! cmp {
        (
            $f:ty, $x:ident, $y:ident, $apfloat_ty:ident, $sys_available:meta,
            $($unordered_val:expr, $fn:ident);*;
        ) => {
            $(
                let cmp0 = if apfloat_fallback!(
                        $f, $apfloat_ty, $sys_available,
                        |x: FloatTy| x.is_nan() => no_convert,
                        $x
                    ) || apfloat_fallback!(
                        $f, $apfloat_ty, $sys_available,
                        |y: FloatTy| y.is_nan() => no_convert,
                        $y
                    )
                {
                    $unordered_val
                } else if apfloat_fallback!(
                    $f, $apfloat_ty, $sys_available,
                    |x, y| x < y => no_convert,
                    $x, $y
                ) {
                    -1
                } else if apfloat_fallback!(
                    $f, $apfloat_ty, $sys_available,
                    |x, y| x == y => no_convert,
                    $x, $y
                ) {
                    0
                } else {
                    1
                };

                let cmp1 = $fn($x, $y);
                if cmp0 != cmp1 {
                    panic!(
                        "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                        stringify!($fn), $x, $y, cmp0, cmp1
                    );
                }
            )*
        };
    }

    #[test]
    #[cfg(f16_enabled)]
    fn cmp_f16() {
        use compiler_builtins::float::cmp::{
            __eqhf2, __gehf2, __gthf2, __lehf2, __lthf2, __nehf2, __unordhf2,
        };

        fuzz_float_2(N, |x: f16, y: f16| {
            assert_eq!(__unordhf2(x, y) != 0, x.is_nan() || y.is_nan());
            cmp!(f16, x, y, Half, all(),
                1, __lthf2;
                1, __lehf2;
                1, __eqhf2;
                -1, __gehf2;
                -1, __gthf2;
                1, __nehf2;
            );
        });
    }

    #[test]
    fn cmp_f32() {
        use compiler_builtins::float::cmp::{
            __eqsf2, __gesf2, __gtsf2, __lesf2, __ltsf2, __nesf2, __unordsf2,
        };

        fuzz_float_2(N, |x: f32, y: f32| {
            assert_eq!(__unordsf2(x, y) != 0, x.is_nan() || y.is_nan());
            cmp!(f32, x, y, Single, all(),
                1, __ltsf2;
                1, __lesf2;
                1, __eqsf2;
                -1, __gesf2;
                -1, __gtsf2;
                1, __nesf2;
            );
        });
    }

    #[test]
    fn cmp_f64() {
        use compiler_builtins::float::cmp::{
            __eqdf2, __gedf2, __gtdf2, __ledf2, __ltdf2, __nedf2, __unorddf2,
        };

        fuzz_float_2(N, |x: f64, y: f64| {
            assert_eq!(__unorddf2(x, y) != 0, x.is_nan() || y.is_nan());
            cmp!(f64, x, y, Double, all(),
                1, __ltdf2;
                1, __ledf2;
                1, __eqdf2;
                -1, __gedf2;
                -1, __gtdf2;
                1, __nedf2;
            );
        });
    }

    #[test]
    #[cfg(f128_enabled)]
    fn cmp_f128() {
        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
        use compiler_builtins::float::cmp::{
            __eqkf2 as __eqtf2, __gekf2 as __getf2, __gtkf2 as __gttf2, __lekf2 as __letf2,
            __ltkf2 as __lttf2, __nekf2 as __netf2, __unordkf2 as __unordtf2,
        };
        #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
        use compiler_builtins::float::cmp::{
            __eqtf2, __getf2, __gttf2, __letf2, __lttf2, __netf2, __unordtf2,
        };

        fuzz_float_2(N, |x: f128, y: f128| {
            let x_is_nan = apfloat_fallback!(
                f128, Quad, not(feature = "no-sys-f128"),
                |x: FloatTy| x.is_nan() => no_convert,
                x
            );
            let y_is_nan = apfloat_fallback!(
                f128, Quad, not(feature = "no-sys-f128"),
                |x: FloatTy| x.is_nan() => no_convert,
                y
            );

            assert_eq!(__unordtf2(x, y) != 0, x_is_nan || y_is_nan);

            cmp!(f128, x, y, Quad, not(feature = "no-sys-f128"),
                1, __lttf2;
                1, __letf2;
                1, __eqtf2;
                -1, __getf2;
                -1, __gttf2;
                1, __netf2;
            );
        });
    }
}

#[cfg(target_arch = "arm")]
mod float_comparisons_arm {
    use super::*;

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

    #[test]
    fn cmp_f32() {
        use compiler_builtins::float::cmp::{
            __aeabi_fcmpeq, __aeabi_fcmpge, __aeabi_fcmpgt, __aeabi_fcmple, __aeabi_fcmplt,
        };

        fuzz_float_2(N, |x: f32, y: f32| {
            cmp2!(x, y,
                0, x < y, __aeabi_fcmplt;
                0, x <= y, __aeabi_fcmple;
                0, x == y, __aeabi_fcmpeq;
                0, x >= y, __aeabi_fcmpge;
                0, x > y, __aeabi_fcmpgt;
            );
        });
    }

    #[test]
    fn cmp_f64() {
        use compiler_builtins::float::cmp::{
            __aeabi_dcmpeq, __aeabi_dcmpge, __aeabi_dcmpgt, __aeabi_dcmple, __aeabi_dcmplt,
        };

        fuzz_float_2(N, |x: f64, y: f64| {
            cmp2!(x, y,
                0, x < y, __aeabi_dcmplt;
                0, x <= y, __aeabi_dcmple;
                0, x == y, __aeabi_dcmpeq;
                0, x >= y, __aeabi_dcmpge;
                0, x > y, __aeabi_dcmpgt;
            );
        });
    }
}
