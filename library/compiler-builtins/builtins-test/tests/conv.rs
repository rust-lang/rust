#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]
// makes configuration easier
#![allow(unused_macros)]
#![allow(unused_imports)]

use builtins_test::*;
use compiler_builtins::float::Float;
use rustc_apfloat::{Float as _, FloatConvert as _};

mod i_to_f {
    use super::*;

    macro_rules! i_to_f {
        ($f_ty:ty, $apfloat_ty:ident, $sys_available:meta, $($i_ty:ty, $fn:ident);*;) => {
            $(
                #[test]
                fn $fn() {
                    use compiler_builtins::float::conv::$fn;
                    use compiler_builtins::int::Int;

                    fuzz(N, |x: $i_ty| {
                        let f0 = apfloat_fallback!(
                            $f_ty, $apfloat_ty, $sys_available,
                            |x| x as $f_ty;
                            // When the builtin is not available, we need to use a different conversion
                            // method (since apfloat doesn't support `as` casting).
                            |x: $i_ty| {
                                use compiler_builtins::int::MinInt;

                                let apf = if <$i_ty>::SIGNED {
                                    FloatTy::from_i128(x.try_into().unwrap()).value
                                } else {
                                    FloatTy::from_u128(x.try_into().unwrap()).value
                                };

                                <$f_ty>::from_bits(apf.to_bits())
                            },
                            x
                        );
                        let f1: $f_ty = $fn(x);

                        #[cfg($sys_available)] {
                            // This makes sure that the conversion produced the best rounding possible, and does
                            // this independent of `x as $into` rounding correctly.
                            // This assumes that float to integer conversion is correct.
                            let y_minus_ulp = <$f_ty>::from_bits(f1.to_bits().wrapping_sub(1)) as $i_ty;
                            let y = f1 as $i_ty;
                            let y_plus_ulp = <$f_ty>::from_bits(f1.to_bits().wrapping_add(1)) as $i_ty;
                            let error_minus = <$i_ty as Int>::abs_diff(y_minus_ulp, x);
                            let error = <$i_ty as Int>::abs_diff(y, x);
                            let error_plus = <$i_ty as Int>::abs_diff(y_plus_ulp, x);

                            // The first two conditions check that none of the two closest float values are
                            // strictly closer in representation to `x`. The second makes sure that rounding is
                            // towards even significand if two float values are equally close to the integer.
                            if error_minus < error
                                || error_plus < error
                                || ((error_minus == error || error_plus == error)
                                    && ((f0.to_bits() & 1) != 0))
                            {
                                if !cfg!(any(
                                    target_arch = "powerpc",
                                    target_arch = "powerpc64"
                                )) {
                                    panic!(
                                        "incorrect rounding by {}({}): {}, ({}, {}, {}), errors ({}, {}, {})",
                                        stringify!($fn),
                                        x,
                                        f1.to_bits(),
                                        y_minus_ulp,
                                        y,
                                        y_plus_ulp,
                                        error_minus,
                                        error,
                                        error_plus,
                                    );
                                }
                            }
                        }

                        // Test against native conversion. We disable testing on all `x86` because of
                        // rounding bugs with `i686`. `powerpc` also has the same rounding bug.
                        if !Float::eq_repr(f0, f1) && !cfg!(any(
                            target_arch = "x86",
                            target_arch = "powerpc",
                            target_arch = "powerpc64"
                        )) {
                            panic!(
                                "{}({}): std: {:?}, builtins: {:?}",
                                stringify!($fn),
                                x,
                                f0,
                                f1,
                            );
                        }
                    });
                }
            )*
        };
    }

    i_to_f! { f32, Single, all(),
        u32, __floatunsisf;
        i32, __floatsisf;
        u64, __floatundisf;
        i64, __floatdisf;
        u128, __floatuntisf;
        i128, __floattisf;
    }

    i_to_f! { f64, Double, all(),
        u32, __floatunsidf;
        i32, __floatsidf;
        u64, __floatundidf;
        i64, __floatdidf;
        u128, __floatuntidf;
        i128, __floattidf;
    }

    #[cfg(not(feature = "no-f16-f128"))]
    #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
    i_to_f! { f128, Quad, not(feature = "no-sys-f128-int-convert"),
        u32, __floatunsitf;
        i32, __floatsitf;
        u64, __floatunditf;
        i64, __floatditf;
        u128, __floatuntitf;
        i128, __floattitf;
    }

    #[cfg(not(feature = "no-f16-f128"))]
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    i_to_f! { f128, Quad, not(feature = "no-sys-f128-int-convert"),
        u32, __floatunsikf;
        i32, __floatsikf;
        u64, __floatundikf;
        i64, __floatdikf;
        u128, __floatuntikf;
        i128, __floattikf;
    }
}

mod f_to_i {
    use super::*;

    macro_rules! f_to_i {
        ($x:ident, $f_ty:ty, $apfloat_ty:ident, $sys_available:meta, $($i_ty:ty, $fn:ident);*;) => {
            $(
                // it is undefined behavior in the first place to do conversions with NaNs
                if !apfloat_fallback!(
                    $f_ty, $apfloat_ty, $sys_available, |x: FloatTy| x.is_nan() => no_convert, $x
                ) {
                    let conv0 = apfloat_fallback!(
                        $f_ty, $apfloat_ty, $sys_available,
                        // Use an `as` cast when the builtin is available on the system.
                        |x| x as $i_ty;
                        // When the builtin is not available, we need to use a different conversion
                        // method (since apfloat doesn't support `as` casting).
                        |x: $f_ty| {
                            use compiler_builtins::int::MinInt;

                            let apf = FloatTy::from_bits(x.to_bits().into());
                            let bits: usize = <$i_ty>::BITS.try_into().unwrap();

                            let err_fn = || panic!(
                                "Unable to convert value {x:?} to type {}:", stringify!($i_ty)
                            );

                            if <$i_ty>::SIGNED {
                               <$i_ty>::try_from(apf.to_i128(bits).value).ok().unwrap_or_else(err_fn)
                            } else {
                               <$i_ty>::try_from(apf.to_u128(bits).value).ok().unwrap_or_else(err_fn)
                            }
                        },
                        $x
                    );
                    let conv1: $i_ty = $fn($x);
                    if conv0 != conv1 {
                        panic!("{}({:?}): std: {:?}, builtins: {:?}", stringify!($fn), $x, conv0, conv1);
                    }
                }
            )*
        };
    }

    #[test]
    fn f32_to_int() {
        use compiler_builtins::float::conv::{
            __fixsfdi, __fixsfsi, __fixsfti, __fixunssfdi, __fixunssfsi, __fixunssfti,
        };

        fuzz_float(N, |x: f32| {
            f_to_i!(x, f32, Single, all(),
                u32, __fixunssfsi;
                u64, __fixunssfdi;
                u128, __fixunssfti;
                i32, __fixsfsi;
                i64, __fixsfdi;
                i128, __fixsfti;
            );
        });
    }

    #[test]
    fn f64_to_int() {
        use compiler_builtins::float::conv::{
            __fixdfdi, __fixdfsi, __fixdfti, __fixunsdfdi, __fixunsdfsi, __fixunsdfti,
        };

        fuzz_float(N, |x: f64| {
            f_to_i!(x, f64, Double, all(),
                u32, __fixunsdfsi;
                u64, __fixunsdfdi;
                u128, __fixunsdfti;
                i32, __fixdfsi;
                i64, __fixdfdi;
                i128, __fixdfti;
            );
        });
    }

    #[test]
    #[cfg(f128_enabled)]
    fn f128_to_int() {
        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
        use compiler_builtins::float::conv::{
            __fixkfdi as __fixtfdi, __fixkfsi as __fixtfsi, __fixkfti as __fixtfti,
            __fixunskfdi as __fixunstfdi, __fixunskfsi as __fixunstfsi,
            __fixunskfti as __fixunstfti,
        };
        #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
        use compiler_builtins::float::conv::{
            __fixtfdi, __fixtfsi, __fixtfti, __fixunstfdi, __fixunstfsi, __fixunstfti,
        };

        fuzz_float(N, |x: f128| {
            f_to_i!(
                x,
                f128,
                Quad,
                not(feature = "no-sys-f128-int-convert"),
                u32, __fixunstfsi;
                u64, __fixunstfdi;
                u128, __fixunstfti;
                i32, __fixtfsi;
                i64, __fixtfdi;
                i128, __fixtfti;
            );
        });
    }
}

macro_rules! f_to_f {
    (
        $mod:ident,
        $(
            $from_ty:ty => $to_ty:ty,
            $from_ap_ty:ident => $to_ap_ty:ident,
            $fn:ident, $sys_available:meta
        );+;
    ) => {$(
        #[test]
        fn $fn() {
            use compiler_builtins::float::{$mod::$fn, Float};
            use rustc_apfloat::ieee::{$from_ap_ty, $to_ap_ty};

            fuzz_float(N, |x: $from_ty| {
                let tmp0: $to_ty = apfloat_fallback!(
                    $from_ty,
                    $from_ap_ty,
                    $sys_available,
                    |x: $from_ty| x as $to_ty;
                    |x: $from_ty| {
                        let from_apf = FloatTy::from_bits(x.to_bits().into());
                        // Get `value` directly to ignore INVALID_OP
                        let to_apf: $to_ap_ty = from_apf.convert(&mut false).value;
                        <$to_ty>::from_bits(to_apf.to_bits().try_into().unwrap())
                    },
                    x
                );
                let tmp1: $to_ty = $fn(x);

                if !Float::eq_repr(tmp0, tmp1) {
                    panic!(
                        "{}({:?}): std: {:?}, builtins: {:?}",
                        stringify!($fn),
                        x,
                        tmp0,
                        tmp1
                    );
                }
            })
        }
    )+};
}

mod extend {
    use super::*;

    f_to_f! {
        extend,
        f32 => f64, Single => Double, __extendsfdf2, all();
    }

    #[cfg(all(f16_enabled, f128_enabled))]
    #[cfg(not(any(
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "loongarch64"
    )))]
    f_to_f! {
        extend,
        f16 => f32, Half => Single, __extendhfsf2, not(feature = "no-sys-f16");
        f16 => f32, Half => Single, __gnu_h2f_ieee, not(feature = "no-sys-f16");
        f16 => f64, Half => Double, __extendhfdf2, not(feature = "no-sys-f16-f64-convert");
        f16 => f128, Half => Quad, __extendhftf2, not(feature = "no-sys-f16-f128-convert");
        f32 => f128, Single => Quad, __extendsftf2, not(feature = "no-sys-f128");
        f64 => f128, Double => Quad, __extenddftf2, not(feature = "no-sys-f128");
    }

    #[cfg(f128_enabled)]
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    f_to_f! {
        extend,
        // FIXME(#655): `f16` tests disabled until we can bootstrap symbols
        f32 => f128, Single => Quad, __extendsfkf2, not(feature = "no-sys-f128");
        f64 => f128, Double => Quad, __extenddfkf2, not(feature = "no-sys-f128");
    }
}

mod trunc {
    use super::*;

    f_to_f! {
        trunc,
        f64 => f32, Double => Single, __truncdfsf2, all();
    }

    #[cfg(all(f16_enabled, f128_enabled))]
    #[cfg(not(any(
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "loongarch64"
    )))]
    f_to_f! {
        trunc,
        f32 => f16, Single => Half, __truncsfhf2, not(feature = "no-sys-f16");
        f32 => f16, Single => Half, __gnu_f2h_ieee, not(feature = "no-sys-f16");
        f64 => f16, Double => Half, __truncdfhf2, not(feature = "no-sys-f16-f64-convert");
        f128 => f16, Quad => Half, __trunctfhf2, not(feature = "no-sys-f16-f128-convert");
        f128 => f32, Quad => Single, __trunctfsf2, not(feature = "no-sys-f128");
        f128 => f64, Quad => Double, __trunctfdf2, not(feature = "no-sys-f128");
    }

    #[cfg(f128_enabled)]
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    f_to_f! {
        trunc,
        // FIXME(#655): `f16` tests disabled until we can bootstrap symbols
        f128 => f32, Quad => Single, __trunckfsf2, not(feature = "no-sys-f128");
        f128 => f64, Quad => Double, __trunckfdf2, not(feature = "no-sys-f128");
    }
}
