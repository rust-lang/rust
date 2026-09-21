#![allow(unused_macros, unused_features)]
#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::*;

mod int_addsub {
    use super::*;

    macro_rules! sum {
        ($($i:ty, $fn_add:ident, $fn_sub:ident);*;) => {
            $(
                #[test]
                fn $fn_add() {
                    use compiler_builtins::int::addsub::{$fn_add, $fn_sub};

                    fuzz_2(N, |x: $i, y: $i| {
                        let add0 = x.wrapping_add(y);
                        let sub0 = x.wrapping_sub(y);
                        let add1: $i = $fn_add(x, y);
                        let sub1: $i = $fn_sub(x, y);
                        if add0 != add1 {
                            panic!(
                                "{}({}, {}): std: {}, builtins: {}",
                                stringify!($fn_add), x, y, add0, add1
                            );
                        }
                        if sub0 != sub1 {
                            panic!(
                                "{}({}, {}): std: {}, builtins: {}",
                                stringify!($fn_sub), x, y, sub0, sub1
                            );
                        }
                    });
                }
            )*
        };
    }

    macro_rules! overflowing_sum {
        ($($i:ty, $fn_add:ident, $fn_sub:ident);*;) => {
            $(
                #[test]
                fn $fn_add() {
                    use compiler_builtins::int::addsub::{$fn_add, $fn_sub};

                    fuzz_2(N, |x: $i, y: $i| {
                        let (add0, add_o0)= x.overflowing_add(y);
                        let (sub0, sub_o0)= x.overflowing_sub(y);
                        let mut add_o1 = 0;
                        let mut sub_o1 = 0;
                        let add1: $i = $fn_add(x, y, &mut add_o1);
                        let sub1: $i = $fn_sub(x, y, &mut sub_o1);
                        if add0 != add1 || i32::from(add_o0) != add_o1 {
                            panic!(
                                "{}({}, {}): std: {:?}, builtins: {:?}",
                                stringify!($fn_add), x, y, (add0, add_o0) , (add1, add_o1)
                            );
                        }
                        if sub0 != sub1 || i32::from(sub_o0) != sub_o1 {
                            panic!(
                                "{}({}, {}): std: {:?}, builtins: {:?}",
                                stringify!($fn_sub), x, y, (sub0, sub_o0) , (sub1, sub_o1)
                            );
                        }
                    });
                }
            )*
        };
    }

    // Integer addition and subtraction is very simple, so 100 fuzzing passes should be plenty.
    sum! {
        u128, __rust_u128_add, __rust_u128_sub;
        i128, __rust_i128_add, __rust_i128_sub;
    }

    overflowing_sum! {
        u128, __rust_u128_addo, __rust_u128_subo;
        i128, __rust_i128_addo, __rust_i128_subo;
    }
}

macro_rules! float_sum {
    ($($f:ty, $fn_add:ident, $fn_sub:ident, $apfloat_ty:ident, $sys_available:meta);*;) => {
        $(
            #[test]
            fn $fn_add() {
                use core::ops::{Add, Sub};
                use imp::{$fn_add, $fn_sub};
                use compiler_builtins::{assert_biteq, support::Float, support::Hex};

                fuzz_float_2(N, |x: $f, y: $f| {
                    let add0 = apfloat_fallback!($f, $apfloat_ty, $sys_available, Add::add, x, y);
                    let sub0 = apfloat_fallback!($f, $apfloat_ty, $sys_available, Sub::sub, x, y);
                    let add1: $f = $fn_add(x, y);
                    let sub1: $f = $fn_sub(x, y);
                    if !Float::eq_repr(add0, add1) {
                        panic!(
                            "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                            stringify!($fn_add), x, y, add0, add1
                        );
                    }
                    if !Float::eq_repr(sub0, sub1) {
                        panic!(
                            "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                            stringify!($fn_sub), x, y, sub0, sub1
                        );
                    }
                });

                let qnan = <$f as Float>::NAN;
                let snan = <$f as Float>::SNAN;
                let qsnan = <$f as Float>::QSNAN;
                let neg_qnan = <$f as Float>::NEG_NAN;
                let neg_snan = <$f as Float>::NEG_SNAN;
                let neg_qsnan = <$f as Float>::NEG_QSNAN;
                let one = <$f as Float>::ONE;

                let nan_cases = [
                    (qnan, qnan, qnan),
                    (qnan, snan, qnan),
                    (qnan, neg_qnan, qnan),
                    (qnan, neg_snan, qnan),
                    (qnan, one, qnan),
                    (snan, qnan, qsnan),
                    (snan, snan, qsnan),
                    (snan, neg_qnan, qsnan),
                    (snan, neg_snan, qsnan),
                    (snan, one, qsnan),
                    (neg_qnan, qnan, neg_qnan),
                    (neg_qnan, snan, neg_qnan),
                    (neg_qnan, neg_qnan, neg_qnan),
                    (neg_qnan, neg_snan, neg_qnan),
                    (neg_qnan, one, neg_qnan),
                    (neg_snan, qnan, neg_qsnan),
                    (neg_snan, snan, neg_qsnan),
                    (neg_snan, neg_qnan, neg_qsnan),
                    (neg_snan, neg_snan, neg_qsnan),
                    (neg_snan, one, neg_qsnan),
                ];
                // Our semantics are to return a quieted version of the first NaN, which means
                // results are flipped for subtraction when the second input is the NaN.
                let add_cases = [
                    (one, qnan, qnan),
                    (one, snan, qsnan),
                    (one, neg_qnan, neg_qnan),
                    (one, neg_snan, neg_qsnan),
                ];
                let sub_cases = [
                    (one, qnan, neg_qnan),
                    (one, snan, neg_qsnan),
                    (one, neg_qnan, qnan),
                    (one, neg_snan, qsnan),
                ];

                for &(x, y, expected) in nan_cases.iter().chain(add_cases.iter()) {
                    assert_biteq!($fn_add(x, y), expected, "{} + {}", Hex(x), Hex(y));
                }
                for &(x, y, expected) in nan_cases.iter().chain(sub_cases.iter()) {
                    assert_biteq!($fn_sub(x, y), expected, "{} - {}", Hex(x), Hex(y));
                }
            }
        )*
    }
}

#[cfg(not(x86_no_sse2))]
mod float_addsub {
    mod imp {
        #[cfg(f16_enabled)]
        pub use compiler_builtins::float::add::__addhf3;
        pub use compiler_builtins::float::add::{__adddf3, __addsf3};
        #[cfg(f16_enabled)]
        pub use compiler_builtins::float::sub::__subhf3;
        pub use compiler_builtins::float::sub::{__subdf3, __subsf3};
        #[cfg(f128_enabled)]
        cfg_select! {
            any(target_arch = "powerpc", target_arch = "powerpc64") => {
                pub use compiler_builtins::float::add::__addkf3 as __addtf3;
                pub use compiler_builtins::float::sub::__subkf3 as __subtf3;
            }
            _ => {
                pub use compiler_builtins::float::add::__addtf3;
                pub use compiler_builtins::float::sub::__subtf3;
            }
        }
    }

    use super::*;

    #[cfg(f16_enabled)]
    float_sum! {
        f16, __addhf3, __subhf3, Half, all();
    }

    float_sum! {
        f32, __addsf3, __subsf3, Single, all();
        f64, __adddf3, __subdf3, Double, all();
    }

    #[cfg(f128_enabled)]
    float_sum! {
        f128, __addtf3, __subtf3, Quad, not(no_sys_f128);
    }
}
