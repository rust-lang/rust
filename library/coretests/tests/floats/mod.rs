use std::num::FpCategory as Fp;
use std::ops::{Add, Div, Mul, Rem, Sub};

#[doc(hidden)]
trait TestableFloat: Sized {
    /// Unsigned int with the same size, for converting to/from bits.
    type Int;
    /// Set the default tolerance for float comparison based on the type.
    const APPROX: Self;
    /// Allow looser tolerance for f32 on miri
    const POWI_APPROX: Self = Self::APPROX;
    /// Tolerance for `powf` tests; some types need looser bounds
    const POWF_APPROX: Self = Self::APPROX;
    /// Allow looser tolerance for f16
    const _180_TO_RADIANS_APPROX: Self = Self::APPROX;
    /// Allow for looser tolerance for f16
    const PI_TO_DEGREES_APPROX: Self = Self::APPROX;
    /// Tolerance for math tests
    const EXP_APPROX: Self = Self::APPROX;
    const LN_APPROX: Self = Self::APPROX;
    const LOG_APPROX: Self = Self::APPROX;
    const LOG2_APPROX: Self = Self::APPROX;
    const LOG10_APPROX: Self = Self::APPROX;
    const ASINH_APPROX: Self = Self::APPROX;
    const ACOSH_APPROX: Self = Self::APPROX;
    const ATANH_APPROX: Self = Self::APPROX;
    const GAMMA_APPROX: Self = Self::APPROX;
    const GAMMA_APPROX_LOOSE: Self = Self::APPROX;
    const LNGAMMA_APPROX: Self = Self::APPROX;
    const LNGAMMA_APPROX_LOOSE: Self = Self::APPROX;
    const ZERO: Self;
    const ONE: Self;

    const MIN_POSITIVE_NORMAL: Self;
    const MAX_SUBNORMAL: Self;
    /// Smallest number
    const TINY: Self;
    /// Next smallest number
    const TINY_UP: Self;
    /// Exponent = 0b11...10, Significand 0b1111..10. Min val > 0
    const MAX_DOWN: Self;
    /// First pattern over the mantissa
    const NAN_MASK1: Self::Int;
    /// Second pattern over the mantissa
    const NAN_MASK2: Self::Int;
    const EPS_ADD: Self;
    const EPS_MUL: Self;
    const EPS_DIV: Self;
    const RAW_1: Self;
    const RAW_12_DOT_5: Self;
    const RAW_1337: Self;
    const RAW_MINUS_14_DOT_25: Self;
    /// The result of 12.3.mul_add(4.5, 6.7)
    const MUL_ADD_RESULT: Self;
    /// The result of (-12.3).mul_add(-4.5, -6.7)
    const NEG_MUL_ADD_RESULT: Self;
    /// Reciprocal of the maximum val
    const MAX_RECIP: Self;
}

impl TestableFloat for f16 {
    type Int = u16;
    const APPROX: Self = 1e-3;
    const POWF_APPROX: Self = 5e-1;
    const _180_TO_RADIANS_APPROX: Self = 1e-2;
    const PI_TO_DEGREES_APPROX: Self = 0.125;
    const EXP_APPROX: Self = 1e-2;
    const LN_APPROX: Self = 1e-2;
    const LOG_APPROX: Self = 1e-2;
    const LOG2_APPROX: Self = 1e-2;
    const LOG10_APPROX: Self = 1e-2;
    const ASINH_APPROX: Self = 1e-2;
    const ACOSH_APPROX: Self = 1e-2;
    const ATANH_APPROX: Self = 1e-2;
    const GAMMA_APPROX: Self = 1e-2;
    const GAMMA_APPROX_LOOSE: Self = 1e-1;
    const LNGAMMA_APPROX: Self = 1e-2;
    const LNGAMMA_APPROX_LOOSE: Self = 1e-1;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7bfe);
    const NAN_MASK1: Self::Int = 0x02aa;
    const NAN_MASK2: Self::Int = 0x0155;
    const EPS_ADD: Self = if cfg!(miri) { 1e1 } else { 0.0 };
    const EPS_MUL: Self = if cfg!(miri) { 1e3 } else { 0.0 };
    const EPS_DIV: Self = if cfg!(miri) { 1e0 } else { 0.0 };
    const RAW_1: Self = Self::from_bits(0x3c00);
    const RAW_12_DOT_5: Self = Self::from_bits(0x4a40);
    const RAW_1337: Self = Self::from_bits(0x6539);
    const RAW_MINUS_14_DOT_25: Self = Self::from_bits(0xcb20);
    const MUL_ADD_RESULT: Self = 62.031;
    const NEG_MUL_ADD_RESULT: Self = 48.625;
    const MAX_RECIP: Self = 1.526624e-5;
}

impl TestableFloat for f32 {
    type Int = u32;
    const APPROX: Self = 1e-6;
    /// Miri adds some extra errors to float functions; make sure the tests still pass.
    /// These values are purely used as a canary to test against and are thus not a stable guarantee Rust provides.
    /// They serve as a way to get an idea of the real precision of floating point operations on different platforms.
    const POWI_APPROX: Self = if cfg!(miri) { 1e-4 } else { Self::APPROX };
    const POWF_APPROX: Self = if cfg!(miri) { 1e-3 } else { 1e-4 };
    const EXP_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const LN_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const LOG_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const LOG2_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const LOG10_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const ASINH_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const ACOSH_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const ATANH_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const GAMMA_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const GAMMA_APPROX_LOOSE: Self = if cfg!(miri) { 1e-2 } else { 1e-4 };
    const LNGAMMA_APPROX: Self = if cfg!(miri) { 1e-3 } else { Self::APPROX };
    const LNGAMMA_APPROX_LOOSE: Self = if cfg!(miri) { 1e-2 } else { 1e-4 };
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7f7f_fffe);
    const NAN_MASK1: Self::Int = 0x002a_aaaa;
    const NAN_MASK2: Self::Int = 0x0055_5555;
    const EPS_ADD: Self = if cfg!(miri) { 1e-3 } else { 0.0 };
    const EPS_MUL: Self = if cfg!(miri) { 1e-1 } else { 0.0 };
    const EPS_DIV: Self = if cfg!(miri) { 1e-4 } else { 0.0 };
    const RAW_1: Self = Self::from_bits(0x3f800000);
    const RAW_12_DOT_5: Self = Self::from_bits(0x41480000);
    const RAW_1337: Self = Self::from_bits(0x44a72000);
    const RAW_MINUS_14_DOT_25: Self = Self::from_bits(0xc1640000);
    const MUL_ADD_RESULT: Self = 62.05;
    const NEG_MUL_ADD_RESULT: Self = 48.65;
    const MAX_RECIP: Self = 2.938736e-39;
}

impl TestableFloat for f64 {
    type Int = u64;
    const APPROX: Self = 1e-6;
    const GAMMA_APPROX_LOOSE: Self = 1e-4;
    const LNGAMMA_APPROX_LOOSE: Self = 1e-4;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7fef_ffff_ffff_fffe);
    const NAN_MASK1: Self::Int = 0x000a_aaaa_aaaa_aaaa;
    const NAN_MASK2: Self::Int = 0x0005_5555_5555_5555;
    const EPS_ADD: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const EPS_MUL: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const EPS_DIV: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const RAW_1: Self = Self::from_bits(0x3ff0000000000000);
    const RAW_12_DOT_5: Self = Self::from_bits(0x4029000000000000);
    const RAW_1337: Self = Self::from_bits(0x4094e40000000000);
    const RAW_MINUS_14_DOT_25: Self = Self::from_bits(0xc02c800000000000);
    const MUL_ADD_RESULT: Self = 62.050000000000004;
    const NEG_MUL_ADD_RESULT: Self = 48.650000000000006;
    const MAX_RECIP: Self = 5.562684646268003e-309;
}

impl TestableFloat for f128 {
    type Int = u128;
    const APPROX: Self = 1e-9;
    const EXP_APPROX: Self = 1e-12;
    const LN_APPROX: Self = 1e-12;
    const LOG_APPROX: Self = 1e-12;
    const LOG2_APPROX: Self = 1e-12;
    const LOG10_APPROX: Self = 1e-12;
    const ASINH_APPROX: Self = 1e-10;
    const ACOSH_APPROX: Self = 1e-10;
    const ATANH_APPROX: Self = 1e-10;
    const GAMMA_APPROX: Self = 1e-12;
    const GAMMA_APPROX_LOOSE: Self = 1e-10;
    const LNGAMMA_APPROX: Self = 1e-12;
    const LNGAMMA_APPROX_LOOSE: Self = 1e-10;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7ffefffffffffffffffffffffffffffe);
    const NAN_MASK1: Self::Int = 0x0000aaaaaaaaaaaaaaaaaaaaaaaaaaaa;
    const NAN_MASK2: Self::Int = 0x00005555555555555555555555555555;
    const EPS_ADD: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const EPS_MUL: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const EPS_DIV: Self = if cfg!(miri) { 1e-6 } else { 0.0 };
    const RAW_1: Self = Self::from_bits(0x3fff0000000000000000000000000000);
    const RAW_12_DOT_5: Self = Self::from_bits(0x40029000000000000000000000000000);
    const RAW_1337: Self = Self::from_bits(0x40094e40000000000000000000000000);
    const RAW_MINUS_14_DOT_25: Self = Self::from_bits(0xc002c800000000000000000000000000);
    const MUL_ADD_RESULT: Self = 62.0500000000000000000000000000000037;
    const NEG_MUL_ADD_RESULT: Self = 48.6500000000000000000000000000000049;
    const MAX_RECIP: Self = 8.40525785778023376565669454330438228902076605e-4933;
}

/// Determine the tolerance for values of the argument type.
const fn lim_for_ty<T: TestableFloat + Copy>(_x: T) -> T {
    T::APPROX
}

// We have runtime ("rt") and const versions of these macros.

/// Verify that floats are within a tolerance of each other.
macro_rules! assert_approx_eq_rt {
    ($a:expr, $b:expr) => {{ assert_approx_eq_rt!($a, $b, $crate::floats::lim_for_ty($a)) }};
    ($a:expr, $b:expr, $lim:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = (*a - *b).abs();
        assert!(
            diff <= $lim,
            "{a:?} is not approximately equal to {b:?} (threshold {lim:?}, difference {diff:?})",
            lim = $lim
        );
    }};
}
macro_rules! assert_approx_eq_const {
    ($a:expr, $b:expr) => {{ assert_approx_eq_const!($a, $b, $crate::floats::lim_for_ty($a)) }};
    ($a:expr, $b:expr, $lim:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = (*a - *b).abs();
        assert!(diff <= $lim);
    }};
}

/// Verify that floats have the same bitwise representation. Used to avoid the default `0.0 == -0.0`
/// behavior, as well as to ensure exact NaN bitpatterns.
macro_rules! assert_biteq_rt {
    (@inner $left:expr, $right:expr, $msg_sep:literal, $($tt:tt)*) => {{
        let l = $left;
        let r = $right;

        // Hack to coerce left and right to the same type
        let mut _eq_ty = l;
        _eq_ty = r;

        // Hack to get the width from a value
        let bits = (l.to_bits() - l.to_bits()).leading_zeros();
        assert!(
            l.to_bits() == r.to_bits(),
            "{msg}{nl}l: {l:?} ({lb:#0width$x})\nr: {r:?} ({rb:#0width$x})",
            msg = format_args!($($tt)*),
            nl = $msg_sep,
            lb = l.to_bits(),
            rb = r.to_bits(),
            width = ((bits / 4) + 2) as usize,
        );

        if !l.is_nan() && !r.is_nan() {
            // Also check that standard equality holds, since most tests use `assert_biteq` rather
            // than `assert_eq`.
            assert_eq!(l, r);
        }
    }};
    ($left:expr, $right:expr , $($tt:tt)*) => {
        assert_biteq_rt!(@inner $left, $right, "\n", $($tt)*)
    };
    ($left:expr, $right:expr $(,)?) => {
        assert_biteq_rt!(@inner $left, $right, "", "")
    };
}
macro_rules! assert_biteq_const {
    (@inner $left:expr, $right:expr, $msg_sep:literal, $($tt:tt)*) => {{
        let l = $left;
        let r = $right;

        // Hack to coerce left and right to the same type
        let mut _eq_ty = l;
        _eq_ty = r;

        assert!(l.to_bits() == r.to_bits());

        if !l.is_nan() && !r.is_nan() {
            // Also check that standard equality holds, since most tests use `assert_biteq` rather
            // than `assert_eq`.
            assert!(l == r);
        }
    }};
    ($left:expr, $right:expr , $($tt:tt)*) => {
        assert_biteq_const!(@inner $left, $right, "\n", $($tt)*)
    };
    ($left:expr, $right:expr $(,)?) => {
        assert_biteq_const!(@inner $left, $right, "", "")
    };
}

// Use the runtime version by default.
// This way, they can be shadowed by the const versions.
pub(crate) use {assert_approx_eq_rt as assert_approx_eq, assert_biteq_rt as assert_biteq};

// Also make the const version available for re-exports.
#[rustfmt::skip]
pub(crate) use assert_biteq_const;
pub(crate) use assert_approx_eq_const;

/// Generate float tests for all our float types, for compile-time and run-time behavior.
///
/// By default all tests run for all float types. Configuration can be applied via `attrs`.
///
/// ```ignore (this is only a sketch)
/// float_test! {
///     name: fn_name, /* function under test */
///     attrs: {
///         // Apply a configuration to the test for a single type
///         f16: #[cfg(target_has_reliable_f16_math)],
///         // Types can be excluded with `cfg(false)`
///         f64: #[cfg(false)],
///     },
///     test<Float> {
///         /* write tests here, using `Float` as the type */
///     }
/// }
/// ```
macro_rules! float_test {
    (
        name: $name:ident,
        attrs: {
            $(const: #[ $($const_meta:meta),+ ] ,)?
            $(f16: #[ $($f16_meta:meta),+ ] ,)?
            $(const f16: #[ $($f16_const_meta:meta),+ ] ,)?
            $(f32: #[ $($f32_meta:meta),+ ] ,)?
            $(const f32: #[ $($f32_const_meta:meta),+ ] ,)?
            $(f64: #[ $($f64_meta:meta),+ ] ,)?
            $(const f64: #[ $($f64_const_meta:meta),+ ] ,)?
            $(f128: #[ $($f128_meta:meta),+ ] ,)?
            $(const f128: #[ $($f128_const_meta:meta),+ ] ,)?
        },
        test<$fty:ident> $test:block
    ) => {
        mod $name {
            use super::*;

            #[test]
            $( $( #[$f16_meta] )+ )?
            fn test_f16() {
                #[allow(unused_imports)]
                use core::f16::consts;
                type $fty = f16;
                #[allow(unused)]
                const fn flt (x: $fty) -> $fty { x }
                $test
            }

            #[test]
            $( $( #[$f32_meta] )+ )?
            fn test_f32() {
                #[allow(unused_imports)]
                use core::f32::consts;
                type $fty = f32;
                #[allow(unused)]
                const fn flt (x: $fty) -> $fty { x }
                $test
            }

            #[test]
            $( $( #[$f64_meta] )+ )?
            fn test_f64() {
                #[allow(unused_imports)]
                use core::f64::consts;
                type $fty = f64;
                #[allow(unused)]
                const fn flt (x: $fty) -> $fty { x }
                $test
            }

            #[test]
            $( $( #[$f128_meta] )+ )?
            fn test_f128() {
                #[allow(unused_imports)]
                use core::f128::consts;
                type $fty = f128;
                #[allow(unused)]
                const fn flt (x: $fty) -> $fty { x }
                $test
            }

            $( $( #[$const_meta] )+ )?
            mod const_ {
                #[allow(unused)]
                use super::TestableFloat;
                #[allow(unused)]
                use std::num::FpCategory as Fp;
                #[allow(unused)]
                use std::ops::{Add, Div, Mul, Rem, Sub};
                // Shadow the runtime versions of the macro with const-compatible versions.
                #[allow(unused)]
                use $crate::floats::{
                    assert_approx_eq_const as assert_approx_eq,
                    assert_biteq_const as assert_biteq,
                };

                #[test]
                $( $( #[$f16_const_meta] )+ )?
                fn test_f16() {
                    #[allow(unused_imports)]
                    use core::f16::consts;
                    type $fty = f16;
                    #[allow(unused)]
                    const fn flt (x: $fty) -> $fty { x }
                    const { $test }
                }

                #[test]
                $( $( #[$f32_const_meta] )+ )?
                fn test_f32() {
                    #[allow(unused_imports)]
                    use core::f32::consts;
                    type $fty = f32;
                    #[allow(unused)]
                    const fn flt (x: $fty) -> $fty { x }
                    const { $test }
                }

                #[test]
                $( $( #[$f64_const_meta] )+ )?
                fn test_f64() {
                    #[allow(unused_imports)]
                    use core::f64::consts;
                    type $fty = f64;
                    #[allow(unused)]
                    const fn flt (x: $fty) -> $fty { x }
                    const { $test }
                }

                #[test]
                $( $( #[$f128_const_meta] )+ )?
                fn test_f128() {
                    #[allow(unused_imports)]
                    use core::f128::consts;
                    type $fty = f128;
                    #[allow(unused)]
                    const fn flt (x: $fty) -> $fty { x }
                    const { $test }
                }
            }
        }
    };
}

float_test! {
    name: num,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let two: Float = 2.0;
        let ten: Float = 10.0;
        assert_biteq!(ten.add(two), ten + two);
        assert_biteq!(ten.sub(two), ten - two);
        assert_biteq!(ten.mul(two), ten * two);
        assert_biteq!(ten.div(two), ten / two);
    }
}

// FIXME(f16_f128): merge into `num` once the required `fmodl`/`fmodf128` function is available on
// all platforms.
float_test! {
    name: num_rem,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        let two: Float = 2.0;
        let ten: Float = 10.0;
        assert_biteq!(ten.rem(two), ten % two);
    }
}

float_test! {
    name: nan,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        assert!(nan.is_nan());
        assert!(!nan.is_infinite());
        assert!(!nan.is_finite());
        assert!(!nan.is_normal());
        assert!(nan.is_sign_positive());
        assert!(!nan.is_sign_negative());
        assert!(matches!(nan.classify(), Fp::Nan));
        // Ensure the quiet bit is set.
        assert!(nan.to_bits() & (1 << (Float::MANTISSA_DIGITS - 2)) != 0);
    }
}

float_test! {
    name: infinity,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let inf: Float = Float::INFINITY;
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(inf.is_sign_positive());
        assert!(!inf.is_sign_negative());
        assert!(!inf.is_nan());
        assert!(!inf.is_normal());
        assert!(matches!(inf.classify(), Fp::Infinite));
    }
}

float_test! {
    name: neg_infinity,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let neg_inf: Float = Float::NEG_INFINITY;
        assert!(neg_inf.is_infinite());
        assert!(!neg_inf.is_finite());
        assert!(!neg_inf.is_sign_positive());
        assert!(neg_inf.is_sign_negative());
        assert!(!neg_inf.is_nan());
        assert!(!neg_inf.is_normal());
        assert!(matches!(neg_inf.classify(), Fp::Infinite));
    }
}

float_test! {
    name: zero,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(0.0, Float::ZERO);
        assert!(!Float::ZERO.is_infinite());
        assert!(Float::ZERO.is_finite());
        assert!(Float::ZERO.is_sign_positive());
        assert!(!Float::ZERO.is_sign_negative());
        assert!(!Float::ZERO.is_nan());
        assert!(!Float::ZERO.is_normal());
        assert!(matches!(Float::ZERO.classify(), Fp::Zero));
    }
}

float_test! {
    name: neg_zero,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let neg_zero: Float = -0.0;
        assert!(0.0 == neg_zero);
        assert_biteq!(-0.0, neg_zero);
        assert!(!neg_zero.is_infinite());
        assert!(neg_zero.is_finite());
        assert!(!neg_zero.is_sign_positive());
        assert!(neg_zero.is_sign_negative());
        assert!(!neg_zero.is_nan());
        assert!(!neg_zero.is_normal());
        assert!(matches!(neg_zero.classify(), Fp::Zero));
    }
}

float_test! {
    name: one,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(1.0, Float::ONE);
        assert!(!Float::ONE.is_infinite());
        assert!(Float::ONE.is_finite());
        assert!(Float::ONE.is_sign_positive());
        assert!(!Float::ONE.is_sign_negative());
        assert!(!Float::ONE.is_nan());
        assert!(Float::ONE.is_normal());
        assert!(matches!(Float::ONE.classify(), Fp::Normal));
    }
}

float_test! {
    name: is_nan,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let pos: Float = 5.3;
        let neg: Float = -10.732;
        assert!(nan.is_nan());
        assert!(!Float::ZERO.is_nan());
        assert!(!pos.is_nan());
        assert!(!neg.is_nan());
        assert!(!inf.is_nan());
        assert!(!neg_inf.is_nan());
    }
}

float_test! {
    name: is_infinite,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let pos: Float = 42.8;
        let neg: Float = -109.2;
        assert!(!nan.is_infinite());
        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(!Float::ZERO.is_infinite());
        assert!(!pos.is_infinite());
        assert!(!neg.is_infinite());
    }
}

float_test! {
    name: is_finite,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let pos: Float = 42.8;
        let neg: Float = -109.2;
        assert!(!nan.is_finite());
        assert!(!inf.is_finite());
        assert!(!neg_inf.is_finite());
        assert!(Float::ZERO.is_finite());
        assert!(pos.is_finite());
        assert!(neg.is_finite());
    }
}

float_test! {
    name: is_normal,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let neg_zero: Float = -0.0;
        assert!(!nan.is_normal());
        assert!(!inf.is_normal());
        assert!(!neg_inf.is_normal());
        assert!(!Float::ZERO.is_normal());
        assert!(!neg_zero.is_normal());
        assert!(Float::ONE.is_normal());
        assert!(Float::MIN_POSITIVE_NORMAL.is_normal());
        assert!(!Float::MAX_SUBNORMAL.is_normal());
    }
}

float_test! {
    name: classify,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let neg_zero: Float = -0.0;
        assert!(matches!(nan.classify(), Fp::Nan));
        assert!(matches!(inf.classify(), Fp::Infinite));
        assert!(matches!(neg_inf.classify(), Fp::Infinite));
        assert!(matches!(Float::ZERO.classify(), Fp::Zero));
        assert!(matches!(neg_zero.classify(), Fp::Zero));
        assert!(matches!(Float::ONE.classify(), Fp::Normal));
        assert!(matches!(Float::MIN_POSITIVE_NORMAL.classify(), Fp::Normal));
        assert!(matches!(Float::MAX_SUBNORMAL.classify(), Fp::Subnormal));
    }
}

float_test! {
    name: min,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).min(0.0), 0.0);
        assert_biteq!(flt(-0.0).min(-0.0), -0.0);
        assert_biteq!(flt(9.0).min(9.0), 9.0);
        assert_biteq!(flt(-9.0).min(0.0), -9.0);
        assert_biteq!(flt(0.0).min(9.0), 0.0);
        assert_biteq!(flt(-0.0).min(9.0), -0.0);
        assert_biteq!(flt(-0.0).min(-9.0), -9.0);
        assert_biteq!(Float::INFINITY.min(9.0), 9.0);
        assert_biteq!(flt(9.0).min(Float::INFINITY), 9.0);
        assert_biteq!(Float::INFINITY.min(-9.0), -9.0);
        assert_biteq!(flt(-9.0).min(Float::INFINITY), -9.0);
        assert_biteq!(Float::NEG_INFINITY.min(9.0), Float::NEG_INFINITY);
        assert_biteq!(flt(9.0).min(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NEG_INFINITY.min(-9.0), Float::NEG_INFINITY);
        assert_biteq!(flt(-9.0).min(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NAN.min(9.0), 9.0);
        assert_biteq!(Float::NAN.min(-9.0), -9.0);
        assert_biteq!(flt(9.0).min(Float::NAN), 9.0);
        assert_biteq!(flt(-9.0).min(Float::NAN), -9.0);
        assert!(Float::NAN.min(Float::NAN).is_nan());
    }
}

float_test! {
    name: max,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).max(0.0), 0.0);
        assert_biteq!(flt(-0.0).max(-0.0), -0.0);
        assert_biteq!(flt(9.0).max(9.0), 9.0);
        assert_biteq!(flt(-9.0).max(0.0), 0.0);
        assert_biteq!(flt(-9.0).max(-0.0), -0.0);
        assert_biteq!(flt(0.0).max(9.0), 9.0);
        assert_biteq!(flt(0.0).max(-9.0), 0.0);
        assert_biteq!(flt(-0.0).max(-9.0), -0.0);
        assert_biteq!(Float::INFINITY.max(9.0), Float::INFINITY);
        assert_biteq!(flt(9.0).max(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::INFINITY.max(-9.0), Float::INFINITY);
        assert_biteq!(flt(-9.0).max(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.max(9.0), 9.0);
        assert_biteq!(flt(9.0).max(Float::NEG_INFINITY), 9.0);
        assert_biteq!(Float::NEG_INFINITY.max(-9.0), -9.0);
        assert_biteq!(flt(-9.0).max(Float::NEG_INFINITY), -9.0);
        assert_biteq!(Float::NAN.max(9.0), 9.0);
        assert_biteq!(Float::NAN.max(-9.0), -9.0);
        assert_biteq!(flt(9.0).max(Float::NAN), 9.0);
        assert_biteq!(flt(-9.0).max(Float::NAN), -9.0);
        assert!(Float::NAN.max(Float::NAN).is_nan());
    }
}

float_test! {
    name: minimum,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).minimum(0.0), 0.0);
        assert_biteq!(flt(-0.0).minimum(0.0), -0.0);
        assert_biteq!(flt(-0.0).minimum(-0.0), -0.0);
        assert_biteq!(flt(9.0).minimum(9.0), 9.0);
        assert_biteq!(flt(-9.0).minimum(0.0), -9.0);
        assert_biteq!(flt(0.0).minimum(9.0), 0.0);
        assert_biteq!(flt(-0.0).minimum(9.0), -0.0);
        assert_biteq!(flt(-0.0).minimum(-9.0), -9.0);
        assert_biteq!(Float::INFINITY.minimum(9.0), 9.0);
        assert_biteq!(flt(9.0).minimum(Float::INFINITY), 9.0);
        assert_biteq!(Float::INFINITY.minimum(-9.0), -9.0);
        assert_biteq!(flt(-9.0).minimum(Float::INFINITY), -9.0);
        assert_biteq!(Float::NEG_INFINITY.minimum(9.0), Float::NEG_INFINITY);
        assert_biteq!(flt(9.0).minimum(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NEG_INFINITY.minimum(-9.0), Float::NEG_INFINITY);
        assert_biteq!(flt(-9.0).minimum(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert!(Float::NAN.minimum(9.0).is_nan());
        assert!(Float::NAN.minimum(-9.0).is_nan());
        assert!(flt(9.0).minimum(Float::NAN).is_nan());
        assert!(flt(-9.0).minimum(Float::NAN).is_nan());
        assert!(Float::NAN.minimum(Float::NAN).is_nan());
    }
}

float_test! {
    name: maximum,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).maximum(0.0), 0.0);
        assert_biteq!(flt(-0.0).maximum(0.0), 0.0);
        assert_biteq!(flt(-0.0).maximum(-0.0), -0.0);
        assert_biteq!(flt(9.0).maximum(9.0), 9.0);
        assert_biteq!(flt(-9.0).maximum(0.0), 0.0);
        assert_biteq!(flt(-9.0).maximum(-0.0), -0.0);
        assert_biteq!(flt(0.0).maximum(9.0), 9.0);
        assert_biteq!(flt(0.0).maximum(-9.0), 0.0);
        assert_biteq!(flt(-0.0).maximum(-9.0), -0.0);
        assert_biteq!(Float::INFINITY.maximum(9.0), Float::INFINITY);
        assert_biteq!(flt(9.0).maximum(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::INFINITY.maximum(-9.0), Float::INFINITY);
        assert_biteq!(flt(-9.0).maximum(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.maximum(9.0), 9.0);
        assert_biteq!(flt(9.0).maximum(Float::NEG_INFINITY), 9.0);
        assert_biteq!(Float::NEG_INFINITY.maximum(-9.0), -9.0);
        assert_biteq!(flt(-9.0).maximum(Float::NEG_INFINITY), -9.0);
        assert!(Float::NAN.maximum(9.0).is_nan());
        assert!(Float::NAN.maximum(-9.0).is_nan());
        assert!(flt(9.0).maximum(Float::NAN).is_nan());
        assert!(flt(-9.0).maximum(Float::NAN).is_nan());
        assert!(Float::NAN.maximum(Float::NAN).is_nan());
    }
}

float_test! {
    name: midpoint,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.5).midpoint(0.5), 0.5);
        assert_biteq!(flt(0.5).midpoint(2.5), 1.5);
        assert_biteq!(flt(3.0).midpoint(4.0), 3.5);
        assert_biteq!(flt(-3.0).midpoint(4.0), 0.5);
        assert_biteq!(flt(3.0).midpoint(-4.0), -0.5);
        assert_biteq!(flt(-3.0).midpoint(-4.0), -3.5);
        assert_biteq!(flt(0.0).midpoint(0.0), 0.0);
        assert_biteq!(flt(-0.0).midpoint(-0.0), -0.0);
        assert_biteq!(flt(-5.0).midpoint(5.0), 0.0);
        assert_biteq!(Float::MAX.midpoint(Float::MIN), 0.0);
        assert_biteq!(Float::MIN.midpoint(Float::MAX), 0.0);
        assert_biteq!(Float::MAX.midpoint(Float::MIN_POSITIVE), Float::MAX / 2.);
        assert_biteq!((-Float::MAX).midpoint(Float::MIN_POSITIVE), -Float::MAX / 2.);
        assert_biteq!(Float::MAX.midpoint(-Float::MIN_POSITIVE), Float::MAX / 2.);
        assert_biteq!((-Float::MAX).midpoint(-Float::MIN_POSITIVE), -Float::MAX / 2.);
        assert_biteq!((Float::MIN_POSITIVE).midpoint(Float::MAX), Float::MAX / 2.);
        assert_biteq!((Float::MIN_POSITIVE).midpoint(-Float::MAX), -Float::MAX / 2.);
        assert_biteq!((-Float::MIN_POSITIVE).midpoint(Float::MAX), Float::MAX / 2.);
        assert_biteq!((-Float::MIN_POSITIVE).midpoint(-Float::MAX), -Float::MAX / 2.);
        assert_biteq!(Float::MAX.midpoint(Float::MAX), Float::MAX);
        assert_biteq!(
            (Float::MIN_POSITIVE).midpoint(Float::MIN_POSITIVE),
            Float::MIN_POSITIVE
        );
        assert_biteq!(
            (-Float::MIN_POSITIVE).midpoint(-Float::MIN_POSITIVE),
            -Float::MIN_POSITIVE
        );
        assert_biteq!(Float::MAX.midpoint(5.0), Float::MAX / 2.0 + 2.5);
        assert_biteq!(Float::MAX.midpoint(-5.0), Float::MAX / 2.0 - 2.5);
        assert_biteq!(Float::INFINITY.midpoint(Float::INFINITY), Float::INFINITY);
        assert_biteq!(
            Float::NEG_INFINITY.midpoint(Float::NEG_INFINITY),
            Float::NEG_INFINITY
        );
        assert!(Float::NEG_INFINITY.midpoint(Float::INFINITY).is_nan());
        assert!(Float::INFINITY.midpoint(Float::NEG_INFINITY).is_nan());
        assert!(Float::NAN.midpoint(1.0).is_nan());
        assert!(flt(1.0).midpoint(Float::NAN).is_nan());
        assert!(Float::NAN.midpoint(Float::NAN).is_nan());
    }
}

// Separate test since the `for` loops cannot be run in `const`.
float_test! {
    name: midpoint_large_magnitude,
    attrs: {
        const: #[cfg(false)],
        // FIXME(f16_f128): `powi` does not work in Miri for these types
        f16: #[cfg(all(not(miri), target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        // test if large differences in magnitude are still correctly computed.
        // NOTE: that because of how small x and y are, x + y can never overflow
        // so (x + y) / 2.0 is always correct
        // in particular, `2.pow(i)` will  never be at the max exponent, so it could
        // be safely doubled, while j is significantly smaller.
        for i in Float::MAX_EXP.saturating_sub(64)..Float::MAX_EXP {
            for j in 0..64u8 {
                let large = flt(2.0).powi(i);
                // a much smaller number, such that there is no chance of overflow to test
                // potential double rounding in midpoint's implementation.
                let small = flt(2.0).powi(Float::MAX_EXP - 1)
                    * Float::EPSILON
                    * Float::from(j);

                let naive = (large + small) / 2.0;
                let midpoint = large.midpoint(small);

                assert_biteq!(naive, midpoint);
            }
        }
    }
}

float_test! {
    name: abs,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(Float::INFINITY.abs(), Float::INFINITY);
        assert_biteq!(Float::ONE.abs(), Float::ONE);
        assert_biteq!(Float::ZERO.abs(), Float::ZERO);
        assert_biteq!((-Float::ZERO).abs(), Float::ZERO);
        assert_biteq!((-Float::ONE).abs(), Float::ONE);
        assert_biteq!(Float::NEG_INFINITY.abs(), Float::INFINITY);
        assert_biteq!((Float::ONE / Float::NEG_INFINITY).abs(), Float::ZERO);
        assert!(Float::NAN.abs().is_nan());
    }
}

float_test! {
    name: copysign,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).copysign(-2.0), -1.0);
        assert_biteq!(flt(-1.0).copysign(2.0), 1.0);
        assert_biteq!(Float::INFINITY.copysign(-0.0), Float::NEG_INFINITY);
        assert_biteq!(Float::NEG_INFINITY.copysign(0.0), Float::INFINITY);
    }
}

float_test! {
    name: rem_euclid,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert!(Float::INFINITY.rem_euclid(42.0).is_nan());
        assert_biteq!(flt(42.0).rem_euclid(Float::INFINITY), 42.0);
        assert!(flt(42.0).rem_euclid(Float::NAN).is_nan());
        assert!(Float::INFINITY.rem_euclid(Float::INFINITY).is_nan());
        assert!(Float::INFINITY.rem_euclid(Float::NAN).is_nan());
        assert!(Float::NAN.rem_euclid(Float::INFINITY).is_nan());
    }
}

float_test! {
    name: div_euclid,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(42.0).div_euclid(Float::INFINITY), 0.0);
        assert!(flt(42.0).div_euclid(Float::NAN).is_nan());
        assert!(Float::INFINITY.div_euclid(Float::INFINITY).is_nan());
        assert!(Float::INFINITY.div_euclid(Float::NAN).is_nan());
        assert!(Float::NAN.div_euclid(Float::INFINITY).is_nan());
    }
}

float_test! {
    name: floor,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).floor(), 1.0);
        assert_biteq!(flt(1.3).floor(), 1.0);
        assert_biteq!(flt(1.5).floor(), 1.0);
        assert_biteq!(flt(1.7).floor(), 1.0);
        assert_biteq!(flt(0.5).floor(), 0.0);
        assert_biteq!(flt(0.0).floor(), 0.0);
        assert_biteq!(flt(-0.0).floor(), -0.0);
        assert_biteq!(flt(-0.5).floor(), -1.0);
        assert_biteq!(flt(-1.0).floor(), -1.0);
        assert_biteq!(flt(-1.3).floor(), -2.0);
        assert_biteq!(flt(-1.5).floor(), -2.0);
        assert_biteq!(flt(-1.7).floor(), -2.0);
        assert_biteq!(Float::MAX.floor(), Float::MAX);
        assert_biteq!(Float::MIN.floor(), Float::MIN);
        assert_biteq!(Float::MIN_POSITIVE.floor(), 0.0);
        assert_biteq!((-Float::MIN_POSITIVE).floor(), -1.0);
        assert!(Float::NAN.floor().is_nan());
        assert_biteq!(Float::INFINITY.floor(), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.floor(), Float::NEG_INFINITY);
    }
}

float_test! {
    name: ceil,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).ceil(), 1.0);
        assert_biteq!(flt(1.3).ceil(), 2.0);
        assert_biteq!(flt(1.5).ceil(), 2.0);
        assert_biteq!(flt(1.7).ceil(), 2.0);
        assert_biteq!(flt(0.5).ceil(), 1.0);
        assert_biteq!(flt(0.0).ceil(), 0.0);
        assert_biteq!(flt(-0.0).ceil(), -0.0);
        assert_biteq!(flt(-0.5).ceil(), -0.0);
        assert_biteq!(flt(-1.0).ceil(), -1.0);
        assert_biteq!(flt(-1.3).ceil(), -1.0);
        assert_biteq!(flt(-1.5).ceil(), -1.0);
        assert_biteq!(flt(-1.7).ceil(), -1.0);
        assert_biteq!(Float::MAX.ceil(), Float::MAX);
        assert_biteq!(Float::MIN.ceil(), Float::MIN);
        assert_biteq!(Float::MIN_POSITIVE.ceil(), 1.0);
        assert_biteq!((-Float::MIN_POSITIVE).ceil(), -0.0);
        assert!(Float::NAN.ceil().is_nan());
        assert_biteq!(Float::INFINITY.ceil(), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.ceil(), Float::NEG_INFINITY);
    }
}

float_test! {
    name: round,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(2.5).round(), 3.0);
        assert_biteq!(flt(1.0).round(), 1.0);
        assert_biteq!(flt(1.3).round(), 1.0);
        assert_biteq!(flt(1.5).round(), 2.0);
        assert_biteq!(flt(1.7).round(), 2.0);
        assert_biteq!(flt(0.5).round(), 1.0);
        assert_biteq!(flt(0.0).round(), 0.0);
        assert_biteq!(flt(-0.0).round(), -0.0);
        assert_biteq!(flt(-0.5).round(), -1.0);
        assert_biteq!(flt(-1.0).round(), -1.0);
        assert_biteq!(flt(-1.3).round(), -1.0);
        assert_biteq!(flt(-1.5).round(), -2.0);
        assert_biteq!(flt(-1.7).round(), -2.0);
        assert_biteq!(Float::MAX.round(), Float::MAX);
        assert_biteq!(Float::MIN.round(), Float::MIN);
        assert_biteq!(Float::MIN_POSITIVE.round(), 0.0);
        assert_biteq!((-Float::MIN_POSITIVE).round(), -0.0);
        assert!(Float::NAN.round().is_nan());
        assert_biteq!(Float::INFINITY.round(), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.round(), Float::NEG_INFINITY);
    }
}

float_test! {
    name: round_ties_even,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(2.5).round_ties_even(), 2.0);
        assert_biteq!(flt(1.0).round_ties_even(), 1.0);
        assert_biteq!(flt(1.3).round_ties_even(), 1.0);
        assert_biteq!(flt(1.5).round_ties_even(), 2.0);
        assert_biteq!(flt(1.7).round_ties_even(), 2.0);
        assert_biteq!(flt(0.5).round_ties_even(), 0.0);
        assert_biteq!(flt(0.0).round_ties_even(), 0.0);
        assert_biteq!(flt(-0.0).round_ties_even(), -0.0);
        assert_biteq!(flt(-0.5).round_ties_even(), -0.0);
        assert_biteq!(flt(-1.0).round_ties_even(), -1.0);
        assert_biteq!(flt(-1.3).round_ties_even(), -1.0);
        assert_biteq!(flt(-1.5).round_ties_even(), -2.0);
        assert_biteq!(flt(-1.7).round_ties_even(), -2.0);
        assert_biteq!(Float::MAX.round_ties_even(), Float::MAX);
        assert_biteq!(Float::MIN.round_ties_even(), Float::MIN);
        assert_biteq!(Float::MIN_POSITIVE.round_ties_even(), 0.0);
        assert_biteq!((-Float::MIN_POSITIVE).round_ties_even(), -0.0);
        assert!(Float::NAN.round_ties_even().is_nan());
        assert_biteq!(Float::INFINITY.round_ties_even(), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.round_ties_even(), Float::NEG_INFINITY);
    }
}

float_test! {
    name: trunc,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).trunc(), 1.0);
        assert_biteq!(flt(1.3).trunc(), 1.0);
        assert_biteq!(flt(1.5).trunc(), 1.0);
        assert_biteq!(flt(1.7).trunc(), 1.0);
        assert_biteq!(flt(0.5).trunc(), 0.0);
        assert_biteq!(flt(0.0).trunc(), 0.0);
        assert_biteq!(flt(-0.0).trunc(), -0.0);
        assert_biteq!(flt(-0.5).trunc(), -0.0);
        assert_biteq!(flt(-1.0).trunc(), -1.0);
        assert_biteq!(flt(-1.3).trunc(), -1.0);
        assert_biteq!(flt(-1.5).trunc(), -1.0);
        assert_biteq!(flt(-1.7).trunc(), -1.0);
        assert_biteq!(Float::MAX.trunc(), Float::MAX);
        assert_biteq!(Float::MIN.trunc(), Float::MIN);
        assert_biteq!(Float::MIN_POSITIVE.trunc(), 0.0);
        assert_biteq!((-Float::MIN_POSITIVE).trunc(), -0.0);
        assert!(Float::NAN.trunc().is_nan());
        assert_biteq!(Float::INFINITY.trunc(), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.trunc(), Float::NEG_INFINITY);
    }
}

float_test! {
    name: fract,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).fract(), 0.0);
        assert_approx_eq!(flt(1.3).fract(), 0.3); // rounding differs between float types
        assert_biteq!(flt(1.5).fract(), 0.5);
        assert_approx_eq!(flt(1.7).fract(), 0.7);
        assert_biteq!(flt(0.5).fract(), 0.5);
        assert_biteq!(flt(0.0).fract(), 0.0);
        assert_biteq!(flt(-0.0).fract(), 0.0);
        assert_biteq!(flt(-0.5).fract(), -0.5);
        assert_biteq!(flt(-1.0).fract(), 0.0);
        assert_approx_eq!(flt(-1.3).fract(), -0.3); // rounding differs between float types
        assert_biteq!(flt(-1.5).fract(), -0.5);
        assert_approx_eq!(flt(-1.7).fract(), -0.7);
        assert_biteq!(Float::MAX.fract(), 0.0);
        assert_biteq!(Float::MIN.fract(), 0.0);
        assert_biteq!(Float::MIN_POSITIVE.fract(), Float::MIN_POSITIVE);
        assert!(Float::MIN_POSITIVE.fract().is_sign_positive());
        assert_biteq!((-Float::MIN_POSITIVE).fract(), -Float::MIN_POSITIVE);
        assert!((-Float::MIN_POSITIVE).fract().is_sign_negative());
        assert!(Float::NAN.fract().is_nan());
        assert!(Float::INFINITY.fract().is_nan());
        assert!(Float::NEG_INFINITY.fract().is_nan());
    }
}

float_test! {
    name: signum,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(Float::INFINITY.signum(), Float::ONE);
        assert_biteq!(Float::ONE.signum(), Float::ONE);
        assert_biteq!(Float::ZERO.signum(), Float::ONE);
        assert_biteq!((-Float::ZERO).signum(), -Float::ONE);
        assert_biteq!((-Float::ONE).signum(), -Float::ONE);
        assert_biteq!(Float::NEG_INFINITY.signum(), -Float::ONE);
        assert_biteq!((Float::ONE / Float::NEG_INFINITY).signum(), -Float::ONE);
        assert!(Float::NAN.signum().is_nan());
    }
}

float_test! {
    name: is_sign_positive,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert!(Float::INFINITY.is_sign_positive());
        assert!(Float::ONE.is_sign_positive());
        assert!(Float::ZERO.is_sign_positive());
        assert!(!(-Float::ZERO).is_sign_positive());
        assert!(!(-Float::ONE).is_sign_positive());
        assert!(!Float::NEG_INFINITY.is_sign_positive());
        assert!(!(Float::ONE / Float::NEG_INFINITY).is_sign_positive());
        assert!(Float::NAN.is_sign_positive());
        assert!(!(-Float::NAN).is_sign_positive());
    }
}

float_test! {
    name: is_sign_negative,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert!(!Float::INFINITY.is_sign_negative());
        assert!(!Float::ONE.is_sign_negative());
        assert!(!Float::ZERO.is_sign_negative());
        assert!((-Float::ZERO).is_sign_negative());
        assert!((-Float::ONE).is_sign_negative());
        assert!(Float::NEG_INFINITY.is_sign_negative());
        assert!((Float::ONE / Float::NEG_INFINITY).is_sign_negative());
        assert!(!Float::NAN.is_sign_negative());
        assert!((-Float::NAN).is_sign_negative());
    }
}

float_test! {
    name: next_up,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(Float::NEG_INFINITY.next_up(), Float::MIN);
        assert_biteq!(Float::MIN.next_up(), -Float::MAX_DOWN);
        assert_biteq!((-Float::ONE - Float::EPSILON).next_up(), -Float::ONE);
        assert_biteq!((-Float::MIN_POSITIVE_NORMAL).next_up(), -Float::MAX_SUBNORMAL);
        assert_biteq!((-Float::TINY_UP).next_up(), -Float::TINY);
        assert_biteq!((-Float::TINY).next_up(), -Float::ZERO);
        assert_biteq!((-Float::ZERO).next_up(), Float::TINY);
        assert_biteq!(Float::ZERO.next_up(), Float::TINY);
        assert_biteq!(Float::TINY.next_up(), Float::TINY_UP);
        assert_biteq!(Float::MAX_SUBNORMAL.next_up(), Float::MIN_POSITIVE_NORMAL);
        assert_biteq!(Float::ONE.next_up(), 1.0 + Float::EPSILON);
        assert_biteq!(Float::MAX.next_up(), Float::INFINITY);
        assert_biteq!(Float::INFINITY.next_up(), Float::INFINITY);

        // Check that NaNs roundtrip.
        let nan0 = Float::NAN;
        let nan1 = Float::from_bits(Float::NAN.to_bits() ^ Float::NAN_MASK1);
        let nan2 = Float::from_bits(Float::NAN.to_bits() ^ Float::NAN_MASK2);
        assert_biteq!(nan0.next_up(), nan0);
        assert_biteq!(nan1.next_up(), nan1);
        assert_biteq!(nan2.next_up(), nan2);
    }
}

float_test! {
    name: next_down,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(Float::NEG_INFINITY.next_down(), Float::NEG_INFINITY);
        assert_biteq!(Float::MIN.next_down(), Float::NEG_INFINITY);
        assert_biteq!((-Float::MAX_DOWN).next_down(), Float::MIN);
        assert_biteq!((-Float::ONE).next_down(), -1.0 - Float::EPSILON);
        assert_biteq!((-Float::MAX_SUBNORMAL).next_down(), -Float::MIN_POSITIVE_NORMAL);
        assert_biteq!((-Float::TINY).next_down(), -Float::TINY_UP);
        assert_biteq!((-Float::ZERO).next_down(), -Float::TINY);
        assert_biteq!((Float::ZERO).next_down(), -Float::TINY);
        assert_biteq!(Float::TINY.next_down(), Float::ZERO);
        assert_biteq!(Float::TINY_UP.next_down(), Float::TINY);
        assert_biteq!(Float::MIN_POSITIVE_NORMAL.next_down(), Float::MAX_SUBNORMAL);
        assert_biteq!((1.0 + Float::EPSILON).next_down(), Float::ONE);
        assert_biteq!(Float::MAX.next_down(), Float::MAX_DOWN);
        assert_biteq!(Float::INFINITY.next_down(), Float::MAX);

        // Check that NaNs roundtrip.
        let nan0 = Float::NAN;
        let nan1 = Float::from_bits(Float::NAN.to_bits() ^ Float::NAN_MASK1);
        let nan2 = Float::from_bits(Float::NAN.to_bits() ^ Float::NAN_MASK2);
        assert_biteq!(nan0.next_down(), nan0);
        assert_biteq!(nan1.next_down(), nan1);
        assert_biteq!(nan2.next_down(), nan2);
    }
}

float_test! {
    name: sqrt_domain,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        assert!(Float::NAN.sqrt().is_nan());
        assert!(Float::NEG_INFINITY.sqrt().is_nan());
        assert!((-Float::ONE).sqrt().is_nan());
        assert_biteq!((-Float::ZERO).sqrt(), -Float::ZERO);
        assert_biteq!(Float::ZERO.sqrt(), Float::ZERO);
        assert_biteq!(Float::ONE.sqrt(), Float::ONE);
        assert_biteq!(Float::INFINITY.sqrt(), Float::INFINITY);
    }
}

float_test! {
    name: clamp_min_greater_than_max,
    attrs: {
        const: #[cfg(false)],
        f16: #[should_panic, cfg(any(miri, target_has_reliable_f16))],
        f32: #[should_panic],
        f64: #[should_panic],
        f128: #[should_panic, cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let _ = Float::ONE.clamp(3.0, 1.0);
    }
}

float_test! {
    name: clamp_min_is_nan,
    attrs: {
        const: #[cfg(false)],
        f16: #[should_panic, cfg(any(miri, target_has_reliable_f16))],
        f32: #[should_panic],
        f64: #[should_panic],
        f128: #[should_panic, cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let _ = Float::ONE.clamp(Float::NAN, 1.0);
    }
}

float_test! {
    name: clamp_max_is_nan,
    attrs: {
        const: #[cfg(false)],
        f16: #[should_panic, cfg(any(miri, target_has_reliable_f16))],
        f32: #[should_panic],
        f64: #[should_panic],
        f128: #[should_panic, cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let _ = Float::ONE.clamp(3.0, Float::NAN);
    }
}

float_test! {
    name: total_cmp,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        use core::cmp::Ordering;

        const fn quiet_bit_mask() -> <Float as TestableFloat>::Int {
            1 << (Float::MANTISSA_DIGITS - 2)
        }

        const fn q_nan() -> Float {
            Float::from_bits(Float::NAN.to_bits() | quiet_bit_mask())
        }

        assert!(matches!(Float::total_cmp(&-q_nan(), &-q_nan()), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-Float::INFINITY, &-Float::INFINITY), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-Float::MAX, &-Float::MAX), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-2.5, &-2.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-1.0, &-1.0), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-1.5, &-1.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-0.5, &-0.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-Float::MIN_POSITIVE, &-Float::MIN_POSITIVE), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::MAX_SUBNORMAL), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-Float::TINY, &-Float::TINY), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&-0.0, &-0.0), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&0.0, &0.0), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&Float::TINY, &Float::TINY), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::MAX_SUBNORMAL), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&Float::MIN_POSITIVE, &Float::MIN_POSITIVE), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&0.5, &0.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&1.0, &1.0), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&1.5, &1.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&2.5, &2.5), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&Float::MAX, &Float::MAX), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&Float::INFINITY, &Float::INFINITY), Ordering::Equal));
        assert!(matches!(Float::total_cmp(&q_nan(), &q_nan()), Ordering::Equal));

        assert!(matches!(Float::total_cmp(&-Float::INFINITY, &-Float::MAX), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-Float::MAX, &-2.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-2.5, &-1.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-1.5, &-1.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-1.0, &-0.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-0.5, &-Float::MIN_POSITIVE), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-Float::MIN_POSITIVE, &-Float::MAX_SUBNORMAL), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::TINY), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-Float::TINY, &-0.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-0.0, &0.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&0.0, &Float::TINY), Ordering::Less));
        assert!(matches!(Float::total_cmp(&Float::TINY, &Float::MAX_SUBNORMAL), Ordering::Less));
        assert!(matches!(Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::MIN_POSITIVE), Ordering::Less));
        assert!(matches!(Float::total_cmp(&Float::MIN_POSITIVE, &0.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&0.5, &1.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&1.0, &1.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&1.5, &2.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&2.5, &Float::MAX), Ordering::Less));
        assert!(matches!(Float::total_cmp(&Float::MAX, &Float::INFINITY), Ordering::Less));

        assert!(matches!(Float::total_cmp(&-Float::MAX, &-Float::INFINITY), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-2.5, &-Float::MAX), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-1.5, &-2.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-1.0, &-1.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-0.5, &-1.0), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-Float::MIN_POSITIVE, &-0.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::MIN_POSITIVE), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-Float::TINY, &-Float::MAX_SUBNORMAL), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&-0.0, &-Float::TINY), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&0.0, &-0.0), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&Float::TINY, &0.0), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::TINY), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&Float::MIN_POSITIVE, &Float::MAX_SUBNORMAL), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&0.5, &Float::MIN_POSITIVE), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&1.0, &0.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&1.5, &1.0), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&2.5, &1.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&Float::MAX, &2.5), Ordering::Greater));
        assert!(matches!(Float::total_cmp(&Float::INFINITY, &Float::MAX), Ordering::Greater));

        assert!(matches!(Float::total_cmp(&-q_nan(), &-Float::INFINITY), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-Float::MAX), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-2.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-1.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-1.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-0.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-Float::MIN_POSITIVE), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-Float::MAX_SUBNORMAL), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-Float::TINY), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &-0.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &0.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &Float::TINY), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &Float::MAX_SUBNORMAL), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &Float::MIN_POSITIVE), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &0.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &1.0), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &1.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &2.5), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &Float::MAX), Ordering::Less));
        assert!(matches!(Float::total_cmp(&-q_nan(), &Float::INFINITY), Ordering::Less));

    }
}

// FIXME(f16): Tests involving sNaN are disabled because without optimizations, `total_cmp` is
// getting incorrectly lowered to code that includes a `extend`/`trunc` round trip, which quiets
// sNaNs. See: https://github.com/llvm/llvm-project/issues/104915

float_test! {
    name: total_cmp_s_nan,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(miri)],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        use core::cmp::Ordering;

        fn quiet_bit_mask() -> <Float as TestableFloat>::Int {
            1 << (Float::MANTISSA_DIGITS - 2)
        }

        fn q_nan() -> Float {
            Float::from_bits(Float::NAN.to_bits() | quiet_bit_mask())
        }

        fn s_nan() -> Float {
            Float::from_bits((Float::NAN.to_bits() & !quiet_bit_mask()) + 42)
        }
        assert_eq!(Ordering::Equal, Float::total_cmp(&-s_nan(), &-s_nan()));
        assert_eq!(Ordering::Equal, Float::total_cmp(&s_nan(), &s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::INFINITY));
        assert_eq!(Ordering::Less, Float::total_cmp(&Float::INFINITY, &s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&s_nan(), &q_nan()));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-s_nan(), &-q_nan()));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-Float::INFINITY, &-s_nan()));
        assert_eq!(Ordering::Greater, Float::total_cmp(&s_nan(), &Float::INFINITY));
        assert_eq!(Ordering::Greater, Float::total_cmp(&q_nan(), &s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &s_nan()));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::INFINITY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &-0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &Float::INFINITY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-s_nan(), &s_nan()));
    }
}

float_test! {
    name: recip,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let max: Float = Float::MAX;
        assert_biteq!(flt(1.0).recip(), 1.0);
        assert_biteq!(flt(2.0).recip(), 0.5);
        assert_biteq!(flt(-0.4).recip(), -2.5);
        assert_biteq!(flt(0.0).recip(), inf);
        assert!(nan.recip().is_nan());
        assert_biteq!(inf.recip(), 0.0);
        assert_biteq!(neg_inf.recip(), -0.0);
        assert_biteq!(max.recip(), Float::MAX_RECIP);
    }
}

float_test! {
    name: powi,
    attrs: {
        const: #[cfg(false)],
        // FIXME(f16_f128): `powi` does not work in Miri for these types
        f16: #[cfg(all(not(miri), target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_approx_eq!(Float::ONE.powi(1), Float::ONE);
        assert_approx_eq!(flt(-3.1).powi(2), 9.6100000000000005506706202140776519387, Float::POWI_APPROX);
        assert_approx_eq!(flt(5.9).powi(-2), 0.028727377190462507313100483690639638451);
        assert_biteq!(flt(8.3).powi(0), Float::ONE);
        assert!(nan.powi(2).is_nan());
        assert_biteq!(inf.powi(3), inf);
        assert_biteq!(neg_inf.powi(2), inf);
    }
}

float_test! {
    name: powf,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(1.0).powf(1.0), 1.0);
        assert_approx_eq!(flt(3.4).powf(4.5), 246.40818323761892815995637964326426756, Float::POWF_APPROX);
        assert_approx_eq!(flt(2.7).powf(-3.2), 0.041652009108526178281070304373500889273, Float::POWF_APPROX);
        assert_approx_eq!(flt(-3.1).powf(2.0), 9.6100000000000005506706202140776519387, Float::POWF_APPROX);
        assert_approx_eq!(flt(5.9).powf(-2.0), 0.028727377190462507313100483690639638451, Float::POWF_APPROX);
        assert_biteq!(flt(8.3).powf(0.0), 1.0);
        assert!(nan.powf(2.0).is_nan());
        assert_biteq!(inf.powf(2.0), inf);
        assert_biteq!(neg_inf.powf(3.0), neg_inf);
    }
}

float_test! {
    name: exp,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(1.0, flt(0.0).exp());
        assert_approx_eq!(consts::E, flt(1.0).exp(), Float::EXP_APPROX);
        assert_approx_eq!(148.41315910257660342111558004055227962348775, flt(5.0).exp(), Float::EXP_APPROX);

        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let nan: Float = Float::NAN;
        assert_biteq!(inf, inf.exp());
        assert_biteq!(0.0, neg_inf.exp());
        assert!(nan.exp().is_nan());
    }
}

float_test! {
    name: exp2,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(32.0, flt(5.0).exp2());
        assert_biteq!(1.0, flt(0.0).exp2());

        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let nan: Float = Float::NAN;
        assert_biteq!(inf, inf.exp2());
        assert_biteq!(0.0, neg_inf.exp2());
        assert!(nan.exp2().is_nan());
    }
}

float_test! {
    name: ln,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_approx_eq!(flt(1.0).exp().ln(), 1.0, Float::LN_APPROX);
        assert!(nan.ln().is_nan());
        assert_biteq!(inf.ln(), inf);
        assert!(neg_inf.ln().is_nan());
        assert!(flt(-2.3).ln().is_nan());
        assert_biteq!(flt(-0.0).ln(), neg_inf);
        assert_biteq!(flt(0.0).ln(), neg_inf);
        assert_approx_eq!(flt(4.0).ln(), 1.3862943611198906188344642429163531366, Float::LN_APPROX);
    }
}

float_test! {
    name: log,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(10.0).log(10.0), 1.0);
        assert_approx_eq!(flt(2.3).log(3.5), 0.66485771361478710036766645911922010272, Float::LOG_APPROX);
        assert_approx_eq!(flt(1.0).exp().log(flt(1.0).exp()), 1.0, Float::LOG_APPROX);
        assert!(flt(1.0).log(1.0).is_nan());
        assert!(flt(1.0).log(-13.9).is_nan());
        assert!(nan.log(2.3).is_nan());
        assert_biteq!(inf.log(10.0), inf);
        assert!(neg_inf.log(8.8).is_nan());
        assert!(flt(-2.3).log(0.1).is_nan());
        assert_biteq!(flt(-0.0).log(2.0), neg_inf);
        assert_biteq!(flt(0.0).log(7.0), neg_inf);
    }
}

float_test! {
    name: log2,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_approx_eq!(flt(10.0).log2(), 3.32192809488736234787031942948939017, Float::LOG2_APPROX);
        assert_approx_eq!(flt(2.3).log2(), 1.2016338611696504130002982471978765921, Float::LOG2_APPROX);
        assert_approx_eq!(flt(1.0).exp().log2(), 1.4426950408889634073599246810018921381, Float::LOG2_APPROX);
        assert!(nan.log2().is_nan());
        assert_biteq!(inf.log2(), inf);
        assert!(neg_inf.log2().is_nan());
        assert!(flt(-2.3).log2().is_nan());
        assert_biteq!(flt(-0.0).log2(), neg_inf);
        assert_biteq!(flt(0.0).log2(), neg_inf);
    }
}

float_test! {
    name: log10,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(10.0).log10(), 1.0);
        assert_approx_eq!(flt(2.3).log10(), 0.36172783601759284532595218865859309898, Float::LOG10_APPROX);
        assert_approx_eq!(flt(1.0).exp().log10(), 0.43429448190325182765112891891660508222, Float::LOG10_APPROX);
        assert_biteq!(flt(1.0).log10(), 0.0);
        assert!(nan.log10().is_nan());
        assert_biteq!(inf.log10(), inf);
        assert!(neg_inf.log10().is_nan());
        assert!(flt(-2.3).log10().is_nan());
        assert_biteq!(flt(-0.0).log10(), neg_inf);
        assert_biteq!(flt(0.0).log10(), neg_inf);
    }
}

float_test! {
    name: asinh,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).asinh(), 0.0);
        assert_biteq!(flt(-0.0).asinh(), -0.0);

        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let nan: Float = Float::NAN;
        assert_biteq!(inf.asinh(), inf);
        assert_biteq!(neg_inf.asinh(), neg_inf);
        assert!(nan.asinh().is_nan());
        assert!(flt(-0.0).asinh().is_sign_negative());

        // issue 63271
        assert_approx_eq!(flt(2.0).asinh(), 1.443635475178810342493276740273105, Float::ASINH_APPROX);
        assert_approx_eq!(flt(-2.0).asinh(), -1.443635475178810342493276740273105, Float::ASINH_APPROX);

        assert_approx_eq!(flt(-200.0).asinh(), -5.991470797049389, Float::ASINH_APPROX);
        if Float::MAX > flt(66000.0) {
             // regression test for the catastrophic cancellation fixed in 72486
             assert_approx_eq!(flt(-67452098.07139316).asinh(), -18.720075426274544393985484294000831757220, Float::ASINH_APPROX);
        }
    }
}

float_test! {
    name: acosh,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(1.0).acosh(), 0.0);
        assert!(flt(0.999).acosh().is_nan());

        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        let nan: Float = Float::NAN;
        assert_biteq!(inf.acosh(), inf);
        assert!(neg_inf.acosh().is_nan());
        assert!(nan.acosh().is_nan());
        assert_approx_eq!(flt(2.0).acosh(), 1.31695789692481670862504634730796844, Float::ACOSH_APPROX);
        assert_approx_eq!(flt(3.0).acosh(), 1.76274717403908605046521864995958461, Float::ACOSH_APPROX);

        if Float::MAX > flt(66000.0) {
            // test for low accuracy from issue 104548
            assert_approx_eq!(flt(60.0), flt(60.0).cosh().acosh(), Float::ACOSH_APPROX);
        }
    }
}

float_test! {
    name: atanh,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_biteq!(flt(0.0).atanh(), 0.0);
        assert_biteq!(flt(-0.0).atanh(), -0.0);

        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(1.0).atanh(), inf);
        assert_biteq!(flt(-1.0).atanh(), neg_inf);

        let nan: Float = Float::NAN;
        assert!(inf.atanh().is_nan());
        assert!(neg_inf.atanh().is_nan());
        assert!(nan.atanh().is_nan());

        assert_approx_eq!(flt(0.5).atanh(), 0.54930614433405484569762261846126285, Float::ATANH_APPROX);
        assert_approx_eq!(flt(-0.5).atanh(), -0.54930614433405484569762261846126285, Float::ATANH_APPROX);
    }
}

float_test! {
    name: gamma,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_approx_eq!(flt(1.0).gamma(), 1.0, Float::GAMMA_APPROX);
        assert_approx_eq!(flt(2.0).gamma(), 1.0, Float::GAMMA_APPROX);
        assert_approx_eq!(flt(3.0).gamma(), 2.0, Float::GAMMA_APPROX);
        assert_approx_eq!(flt(4.0).gamma(), 6.0, Float::GAMMA_APPROX);
        assert_approx_eq!(flt(5.0).gamma(), 24.0, Float::GAMMA_APPROX_LOOSE);
        assert_approx_eq!(flt(0.5).gamma(), consts::PI.sqrt(), Float::GAMMA_APPROX);
        assert_approx_eq!(flt(-0.5).gamma(), flt(-2.0) * consts::PI.sqrt(), Float::GAMMA_APPROX_LOOSE);
        assert_biteq!(flt(0.0).gamma(), Float::INFINITY);
        assert_biteq!(flt(-0.0).gamma(), Float::NEG_INFINITY);
        assert!(flt(-1.0).gamma().is_nan());
        assert!(flt(-2.0).gamma().is_nan());
        assert!(Float::NAN.gamma().is_nan());
        assert!(Float::NEG_INFINITY.gamma().is_nan());
        assert_biteq!(Float::INFINITY.gamma(), Float::INFINITY);
        assert_biteq!(flt(1760.9).gamma(), Float::INFINITY);
    }
}

float_test! {
    name: ln_gamma,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        assert_approx_eq!(flt(1.0).ln_gamma().0, 0.0, Float::LNGAMMA_APPROX);
        assert_eq!(flt(1.0).ln_gamma().1, 1);
        assert_approx_eq!(flt(2.0).ln_gamma().0, 0.0, Float::LNGAMMA_APPROX);
        assert_eq!(flt(2.0).ln_gamma().1, 1);
        assert_approx_eq!(flt(3.0).ln_gamma().0, flt(2.0).ln(), Float::LNGAMMA_APPROX);
        assert_eq!(flt(3.0).ln_gamma().1, 1);
        assert_approx_eq!(flt(-0.5).ln_gamma().0, (flt(2.0) * consts::PI.sqrt()).ln(), Float::LNGAMMA_APPROX_LOOSE);
        assert_eq!(flt(-0.5).ln_gamma().1, -1);
    }
}

float_test! {
    name: to_degrees,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        let pi: Float = consts::PI;
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(0.0).to_degrees(), 0.0);
        assert_approx_eq!(flt(-5.8).to_degrees(), -332.31552117587745090765431723855668471);
        assert_approx_eq!(pi.to_degrees(), 180.0, Float::PI_TO_DEGREES_APPROX);
        assert!(nan.to_degrees().is_nan());
        assert_biteq!(inf.to_degrees(), inf);
        assert_biteq!(neg_inf.to_degrees(), neg_inf);
        assert_biteq!(flt(1.0).to_degrees(), 57.2957795130823208767981548141051703);
    }
}

float_test! {
    name: to_radians,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(any(miri, target_has_reliable_f128_math))],
    },
    test<Float> {
        let pi: Float = consts::PI;
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(0.0).to_radians(), 0.0);
        assert_approx_eq!(flt(154.6).to_radians(), 2.6982790235832334267135442069489767804);
        assert_approx_eq!(flt(-332.31).to_radians(), -5.7999036373023566567593094812182763013);
        assert_approx_eq!(flt(180.0).to_radians(), pi, Float::_180_TO_RADIANS_APPROX);
        assert!(nan.to_radians().is_nan());
        assert_biteq!(inf.to_radians(), inf);
        assert_biteq!(neg_inf.to_radians(), neg_inf);
    }
}

float_test! {
    name: to_algebraic,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let a: Float = flt(123.0);
        let b: Float = flt(456.0);

        // Check that individual operations match their primitive counterparts.
        //
        // This is a check of current implementations and does NOT imply any form of
        // guarantee about future behavior. The compiler reserves the right to make
        // these operations inexact matches in the future.

        assert_approx_eq!(a.algebraic_add(b), a + b, Float::EPS_ADD);
        assert_approx_eq!(a.algebraic_sub(b), a - b, Float::EPS_ADD);
        assert_approx_eq!(a.algebraic_mul(b), a * b, Float::EPS_MUL);
        assert_approx_eq!(a.algebraic_div(b), a / b, Float::EPS_DIV);
        assert_approx_eq!(a.algebraic_rem(b), a % b, Float::EPS_DIV);
    }
}

float_test! {
    name: to_bits_conv,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(flt(1.0), Float::RAW_1);
        assert_biteq!(flt(12.5), Float::RAW_12_DOT_5);
        assert_biteq!(flt(1337.0), Float::RAW_1337);
        assert_biteq!(flt(-14.25), Float::RAW_MINUS_14_DOT_25);
        assert_biteq!(Float::RAW_1, 1.0);
        assert_biteq!(Float::RAW_12_DOT_5, 12.5);
        assert_biteq!(Float::RAW_1337, 1337.0);
        assert_biteq!(Float::RAW_MINUS_14_DOT_25, -14.25);

        // Check that NaNs roundtrip their bits regardless of signaling-ness
        let masked_nan1 = Float::NAN.to_bits() ^ Float::NAN_MASK1;
        let masked_nan2 = Float::NAN.to_bits() ^ Float::NAN_MASK2;
        assert!(Float::from_bits(masked_nan1).is_nan());
        assert!(Float::from_bits(masked_nan2).is_nan());

        assert_biteq!(Float::from_bits(masked_nan1), Float::from_bits(masked_nan1));
        assert_biteq!(Float::from_bits(masked_nan2), Float::from_bits(masked_nan2));
    }
}

float_test! {
    name: mul_add,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        // FIXME(#140515): mingw has an incorrect fma https://sourceforge.net/p/mingw-w64/bugs/848/
        f32: #[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)],
        f64: #[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        let nan: Float = Float::NAN;
        let inf: Float = Float::INFINITY;
        let neg_inf: Float = Float::NEG_INFINITY;
        assert_biteq!(flt(12.3).mul_add(flt(4.5), flt(6.7)), Float::MUL_ADD_RESULT);
        assert_biteq!((flt(-12.3)).mul_add(flt(-4.5), flt(-6.7)), Float::NEG_MUL_ADD_RESULT);
        assert_biteq!(flt(0.0).mul_add(8.9, 1.2), 1.2);
        assert_biteq!(flt(3.4).mul_add(-0.0, 5.6), 5.6);
        assert!(nan.mul_add(7.8, 9.0).is_nan());
        assert_biteq!(inf.mul_add(7.8, 9.0), inf);
        assert_biteq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
        assert_biteq!(flt(8.9).mul_add(inf, 3.2), inf);
        assert_biteq!((flt(-3.2)).mul_add(2.4, neg_inf), neg_inf);
    }
}

float_test! {
    name: from,
    attrs: {
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(Float::from(false), Float::ZERO);
        assert_biteq!(Float::from(true), Float::ONE);

        assert_biteq!(Float::from(u8::MIN), Float::ZERO);
        assert_biteq!(Float::from(42_u8), 42.0);
        assert_biteq!(Float::from(u8::MAX), 255.0);

        assert_biteq!(Float::from(i8::MIN), -128.0);
        assert_biteq!(Float::from(42_i8), 42.0);
        assert_biteq!(Float::from(i8::MAX), 127.0);
    }
}

float_test! {
    name: from_u16_i16,
    attrs: {
        f16: #[cfg(false)],
        const f16: #[cfg(false)],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(Float::from(u16::MIN), Float::ZERO);
        assert_biteq!(Float::from(42_u16), 42.0);
        assert_biteq!(Float::from(u16::MAX), 65535.0);
        assert_biteq!(Float::from(i16::MIN), -32768.0);
        assert_biteq!(Float::from(42_i16), 42.0);
        assert_biteq!(Float::from(i16::MAX), 32767.0);
    }
}

float_test! {
    name: from_u32_i32,
    attrs: {
        f16: #[cfg(false)],
        const f16: #[cfg(false)],
        f32: #[cfg(false)],
        const f32: #[cfg(false)],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test<Float> {
        assert_biteq!(Float::from(u32::MIN), Float::ZERO);
        assert_biteq!(Float::from(42_u32), 42.0);
        assert_biteq!(Float::from(u32::MAX), 4294967295.0);
        assert_biteq!(Float::from(i32::MIN), -2147483648.0);
        assert_biteq!(Float::from(42_i32), 42.0);
        assert_biteq!(Float::from(i32::MAX), 2147483647.0);
    }
}

// FIXME(f16_f128): Uncomment and adapt these tests once the From<{u64,i64}> impls are added.
// float_test! {
//     name: from_u64_i64,
//     attrs: {
//         f16: #[cfg(false)],
//         f32: #[cfg(false)],
//         f64: #[cfg(false)],
//         f128: #[cfg(any(miri, target_has_reliable_f128))],
//     },
//     test<Float> {
//         assert_biteq!(Float::from(u64::MIN), Float::ZERO);
//         assert_biteq!(Float::from(42_u64), 42.0);
//         assert_biteq!(Float::from(u64::MAX), 18446744073709551615.0);
//         assert_biteq!(Float::from(i64::MIN), -9223372036854775808.0);
//         assert_biteq!(Float::from(42_i64), 42.0);
//         assert_biteq!(Float::from(i64::MAX), 9223372036854775807.0);
//     }
// }

float_test! {
    name: real_consts,
    attrs: {
        // FIXME(f16_f128): add math tests when available
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        let pi: Float = consts::PI;
        assert_approx_eq!(consts::FRAC_PI_2, pi / 2.0);
        assert_approx_eq!(consts::FRAC_PI_3, pi / 3.0, Float::APPROX);
        assert_approx_eq!(consts::FRAC_PI_4, pi / 4.0);
        assert_approx_eq!(consts::FRAC_PI_6, pi / 6.0);
        assert_approx_eq!(consts::FRAC_PI_8, pi / 8.0);
        assert_approx_eq!(consts::FRAC_1_PI, 1.0 / pi);
        assert_approx_eq!(consts::FRAC_2_PI, 2.0 / pi);
        assert_approx_eq!(consts::FRAC_2_SQRT_PI, 2.0 / pi.sqrt());
        assert_approx_eq!(consts::SQRT_2, flt(2.0).sqrt());
        assert_approx_eq!(consts::FRAC_1_SQRT_2, 1.0 / flt(2.0).sqrt());
        assert_approx_eq!(consts::LOG2_E, consts::E.log2());
        assert_approx_eq!(consts::LOG10_E, consts::E.log10());
        assert_approx_eq!(consts::LN_2, flt(2.0).ln());
        assert_approx_eq!(consts::LN_10, flt(10.0).ln(), Float::APPROX);
    }
}
