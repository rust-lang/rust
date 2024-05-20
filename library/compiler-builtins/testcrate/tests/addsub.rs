#![allow(unused_macros)]
#![feature(f128)]
#![feature(f16)]

use testcrate::*;

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
                        let add0 = x.overflowing_add(y);
                        let sub0 = x.overflowing_sub(y);
                        let add1: ($i, bool) = $fn_add(x, y);
                        let sub1: ($i, bool) = $fn_sub(x, y);
                        if add0.0 != add1.0 || add0.1 != add1.1 {
                            panic!(
                                "{}({}, {}): std: {:?}, builtins: {:?}",
                                stringify!($fn_add), x, y, add0, add1
                            );
                        }
                        if sub0.0 != sub1.0 || sub0.1 != sub1.1 {
                            panic!(
                                "{}({}, {}): std: {:?}, builtins: {:?}",
                                stringify!($fn_sub), x, y, sub0, sub1
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
                use compiler_builtins::float::{{add::$fn_add, sub::$fn_sub}, Float};

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
            }
        )*
    }
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
mod float_addsub {
    use super::*;

    float_sum! {
        f32, __addsf3, __subsf3, Single, all();
        f64, __adddf3, __subdf3, Double, all();
    }
}

#[cfg(not(feature = "no-f16-f128"))]
#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
#[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
mod float_addsub_f128 {
    use super::*;

    float_sum! {
        f128, __addtf3, __subtf3, Quad, not(feature = "no-sys-f128");
    }
}

#[cfg(not(feature = "no-f16-f128"))]
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
mod float_addsub_f128_ppc {
    use super::*;

    float_sum! {
        f128, __addkf3, __subkf3, Quad, not(feature = "no-sys-f128");
    }
}

#[cfg(target_arch = "arm")]
mod float_addsub_arm {
    use super::*;

    float_sum! {
        f32, __addsf3vfp, __subsf3vfp, Single, all();
        f64, __adddf3vfp, __subdf3vfp, Double, all();
    }
}
