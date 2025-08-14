use std::num::FpCategory as Fp;
use std::ops::{Add, Div, Mul, Rem, Sub};

trait TestableFloat {
    /// Unsigned int with the same size, for converting to/from bits.
    type Int;
    /// Set the default tolerance for float comparison based on the type.
    const APPROX: Self;
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
}

impl TestableFloat for f16 {
    type Int = u16;
    const APPROX: Self = 1e-3;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7bfe);
    const NAN_MASK1: Self::Int = 0x02aa;
    const NAN_MASK2: Self::Int = 0x0155;
}

impl TestableFloat for f32 {
    type Int = u32;
    const APPROX: Self = 1e-6;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7f7f_fffe);
    const NAN_MASK1: Self::Int = 0x002a_aaaa;
    const NAN_MASK2: Self::Int = 0x0055_5555;
}

impl TestableFloat for f64 {
    type Int = u64;
    const APPROX: Self = 1e-6;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7fef_ffff_ffff_fffe);
    const NAN_MASK1: Self::Int = 0x000a_aaaa_aaaa_aaaa;
    const NAN_MASK2: Self::Int = 0x0005_5555_5555_5555;
}

impl TestableFloat for f128 {
    type Int = u128;
    const APPROX: Self = 1e-9;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN_POSITIVE_NORMAL: Self = Self::MIN_POSITIVE;
    const MAX_SUBNORMAL: Self = Self::MIN_POSITIVE.next_down();
    const TINY: Self = Self::from_bits(0x1);
    const TINY_UP: Self = Self::from_bits(0x2);
    const MAX_DOWN: Self = Self::from_bits(0x7ffefffffffffffffffffffffffffffe);
    const NAN_MASK1: Self::Int = 0x0000aaaaaaaaaaaaaaaaaaaaaaaaaaaa;
    const NAN_MASK2: Self::Int = 0x00005555555555555555555555555555;
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
                type $fty = f16;
                $test
            }

            #[test]
            $( $( #[$f32_meta] )+ )?
            fn test_f32() {
                type $fty = f32;
                $test
            }

            #[test]
            $( $( #[$f64_meta] )+ )?
            fn test_f64() {
                type $fty = f64;
                $test
            }

            #[test]
            $( $( #[$f128_meta] )+ )?
            fn test_f128() {
                type $fty = f128;
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
                    type $fty = f16;
                    const { $test }
                }

                #[test]
                $( $( #[$f32_const_meta] )+ )?
                fn test_f32() {
                    type $fty = f32;
                    const { $test }
                }

                #[test]
                $( $( #[$f64_const_meta] )+ )?
                fn test_f64() {
                    type $fty = f64;
                    const { $test }
                }

                #[test]
                $( $( #[$f128_const_meta] )+ )?
                fn test_f128() {
                    type $fty = f128;
                    const { $test }
                }
            }
        }
    };
}

mod f128;
mod f16;
mod f32;
mod f64;

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
        assert_biteq!((0.0 as Float).min(0.0), 0.0);
        assert_biteq!((-0.0 as Float).min(-0.0), -0.0);
        assert_biteq!((9.0 as Float).min(9.0), 9.0);
        assert_biteq!((-9.0 as Float).min(0.0), -9.0);
        assert_biteq!((0.0 as Float).min(9.0), 0.0);
        assert_biteq!((-0.0 as Float).min(9.0), -0.0);
        assert_biteq!((-0.0 as Float).min(-9.0), -9.0);
        assert_biteq!(Float::INFINITY.min(9.0), 9.0);
        assert_biteq!((9.0 as Float).min(Float::INFINITY), 9.0);
        assert_biteq!(Float::INFINITY.min(-9.0), -9.0);
        assert_biteq!((-9.0 as Float).min(Float::INFINITY), -9.0);
        assert_biteq!(Float::NEG_INFINITY.min(9.0), Float::NEG_INFINITY);
        assert_biteq!((9.0 as Float).min(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NEG_INFINITY.min(-9.0), Float::NEG_INFINITY);
        assert_biteq!((-9.0 as Float).min(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NAN.min(9.0), 9.0);
        assert_biteq!(Float::NAN.min(-9.0), -9.0);
        assert_biteq!((9.0 as Float).min(Float::NAN), 9.0);
        assert_biteq!((-9.0 as Float).min(Float::NAN), -9.0);
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
        assert_biteq!((0.0 as Float).max(0.0), 0.0);
        assert_biteq!((-0.0 as Float).max(-0.0), -0.0);
        assert_biteq!((9.0 as Float).max(9.0), 9.0);
        assert_biteq!((-9.0 as Float).max(0.0), 0.0);
        assert_biteq!((-9.0 as Float).max(-0.0), -0.0);
        assert_biteq!((0.0 as Float).max(9.0), 9.0);
        assert_biteq!((0.0 as Float).max(-9.0), 0.0);
        assert_biteq!((-0.0 as Float).max(-9.0), -0.0);
        assert_biteq!(Float::INFINITY.max(9.0), Float::INFINITY);
        assert_biteq!((9.0 as Float).max(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::INFINITY.max(-9.0), Float::INFINITY);
        assert_biteq!((-9.0 as Float).max(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.max(9.0), 9.0);
        assert_biteq!((9.0 as Float).max(Float::NEG_INFINITY), 9.0);
        assert_biteq!(Float::NEG_INFINITY.max(-9.0), -9.0);
        assert_biteq!((-9.0 as Float).max(Float::NEG_INFINITY), -9.0);
        assert_biteq!(Float::NAN.max(9.0), 9.0);
        assert_biteq!(Float::NAN.max(-9.0), -9.0);
        assert_biteq!((9.0 as Float).max(Float::NAN), 9.0);
        assert_biteq!((-9.0 as Float).max(Float::NAN), -9.0);
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
        assert_biteq!((0.0 as Float).minimum(0.0), 0.0);
        assert_biteq!((-0.0 as Float).minimum(0.0), -0.0);
        assert_biteq!((-0.0 as Float).minimum(-0.0), -0.0);
        assert_biteq!((9.0 as Float).minimum(9.0), 9.0);
        assert_biteq!((-9.0 as Float).minimum(0.0), -9.0);
        assert_biteq!((0.0 as Float).minimum(9.0), 0.0);
        assert_biteq!((-0.0 as Float).minimum(9.0), -0.0);
        assert_biteq!((-0.0 as Float).minimum(-9.0), -9.0);
        assert_biteq!(Float::INFINITY.minimum(9.0), 9.0);
        assert_biteq!((9.0 as Float).minimum(Float::INFINITY), 9.0);
        assert_biteq!(Float::INFINITY.minimum(-9.0), -9.0);
        assert_biteq!((-9.0 as Float).minimum(Float::INFINITY), -9.0);
        assert_biteq!(Float::NEG_INFINITY.minimum(9.0), Float::NEG_INFINITY);
        assert_biteq!((9.0 as Float).minimum(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert_biteq!(Float::NEG_INFINITY.minimum(-9.0), Float::NEG_INFINITY);
        assert_biteq!((-9.0 as Float).minimum(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert!(Float::NAN.minimum(9.0).is_nan());
        assert!(Float::NAN.minimum(-9.0).is_nan());
        assert!((9.0 as Float).minimum(Float::NAN).is_nan());
        assert!((-9.0 as Float).minimum(Float::NAN).is_nan());
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
        assert_biteq!((0.0 as Float).maximum(0.0), 0.0);
        assert_biteq!((-0.0 as Float).maximum(0.0), 0.0);
        assert_biteq!((-0.0 as Float).maximum(-0.0), -0.0);
        assert_biteq!((9.0 as Float).maximum(9.0), 9.0);
        assert_biteq!((-9.0 as Float).maximum(0.0), 0.0);
        assert_biteq!((-9.0 as Float).maximum(-0.0), -0.0);
        assert_biteq!((0.0 as Float).maximum(9.0), 9.0);
        assert_biteq!((0.0 as Float).maximum(-9.0), 0.0);
        assert_biteq!((-0.0 as Float).maximum(-9.0), -0.0);
        assert_biteq!(Float::INFINITY.maximum(9.0), Float::INFINITY);
        assert_biteq!((9.0 as Float).maximum(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::INFINITY.maximum(-9.0), Float::INFINITY);
        assert_biteq!((-9.0 as Float).maximum(Float::INFINITY), Float::INFINITY);
        assert_biteq!(Float::NEG_INFINITY.maximum(9.0), 9.0);
        assert_biteq!((9.0 as Float).maximum(Float::NEG_INFINITY), 9.0);
        assert_biteq!(Float::NEG_INFINITY.maximum(-9.0), -9.0);
        assert_biteq!((-9.0 as Float).maximum(Float::NEG_INFINITY), -9.0);
        assert!(Float::NAN.maximum(9.0).is_nan());
        assert!(Float::NAN.maximum(-9.0).is_nan());
        assert!((9.0 as Float).maximum(Float::NAN).is_nan());
        assert!((-9.0 as Float).maximum(Float::NAN).is_nan());
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
        assert_biteq!((0.5 as Float).midpoint(0.5), 0.5);
        assert_biteq!((0.5 as Float).midpoint(2.5), 1.5);
        assert_biteq!((3.0 as Float).midpoint(4.0), 3.5);
        assert_biteq!((-3.0 as Float).midpoint(4.0), 0.5);
        assert_biteq!((3.0 as Float).midpoint(-4.0), -0.5);
        assert_biteq!((-3.0 as Float).midpoint(-4.0), -3.5);
        assert_biteq!((0.0 as Float).midpoint(0.0), 0.0);
        assert_biteq!((-0.0 as Float).midpoint(-0.0), -0.0);
        assert_biteq!((-5.0 as Float).midpoint(5.0), 0.0);
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
        assert!((1.0 as Float).midpoint(Float::NAN).is_nan());
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
                let large = (2.0 as Float).powi(i);
                // a much smaller number, such that there is no chance of overflow to test
                // potential double rounding in midpoint's implementation.
                let small = (2.0 as Float).powi(Float::MAX_EXP - 1)
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
        assert_biteq!((1.0 as Float).copysign(-2.0), -1.0);
        assert_biteq!((-1.0 as Float).copysign(2.0), 1.0);
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
        assert!(Float::INFINITY.rem_euclid(42.0 as Float).is_nan());
        assert_biteq!((42.0 as Float).rem_euclid(Float::INFINITY), 42.0 as Float);
        assert!((42.0 as Float).rem_euclid(Float::NAN).is_nan());
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
        assert_biteq!((42.0 as Float).div_euclid(Float::INFINITY), 0.0);
        assert!((42.0 as Float).div_euclid(Float::NAN).is_nan());
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
        assert_biteq!((1.0 as Float).floor(), 1.0);
        assert_biteq!((1.3 as Float).floor(), 1.0);
        assert_biteq!((1.5 as Float).floor(), 1.0);
        assert_biteq!((1.7 as Float).floor(), 1.0);
        assert_biteq!((0.5 as Float).floor(), 0.0);
        assert_biteq!((0.0 as Float).floor(), 0.0);
        assert_biteq!((-0.0 as Float).floor(), -0.0);
        assert_biteq!((-0.5 as Float).floor(), -1.0);
        assert_biteq!((-1.0 as Float).floor(), -1.0);
        assert_biteq!((-1.3 as Float).floor(), -2.0);
        assert_biteq!((-1.5 as Float).floor(), -2.0);
        assert_biteq!((-1.7 as Float).floor(), -2.0);
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
        assert_biteq!((1.0 as Float).ceil(), 1.0);
        assert_biteq!((1.3 as Float).ceil(), 2.0);
        assert_biteq!((1.5 as Float).ceil(), 2.0);
        assert_biteq!((1.7 as Float).ceil(), 2.0);
        assert_biteq!((0.5 as Float).ceil(), 1.0);
        assert_biteq!((0.0 as Float).ceil(), 0.0);
        assert_biteq!((-0.0 as Float).ceil(), -0.0);
        assert_biteq!((-0.5 as Float).ceil(), -0.0);
        assert_biteq!((-1.0 as Float).ceil(), -1.0);
        assert_biteq!((-1.3 as Float).ceil(), -1.0);
        assert_biteq!((-1.5 as Float).ceil(), -1.0);
        assert_biteq!((-1.7 as Float).ceil(), -1.0);
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
        assert_biteq!((2.5 as Float).round(), 3.0);
        assert_biteq!((1.0 as Float).round(), 1.0);
        assert_biteq!((1.3 as Float).round(), 1.0);
        assert_biteq!((1.5 as Float).round(), 2.0);
        assert_biteq!((1.7 as Float).round(), 2.0);
        assert_biteq!((0.5 as Float).round(), 1.0);
        assert_biteq!((0.0 as Float).round(), 0.0);
        assert_biteq!((-0.0 as Float).round(), -0.0);
        assert_biteq!((-0.5 as Float).round(), -1.0);
        assert_biteq!((-1.0 as Float).round(), -1.0);
        assert_biteq!((-1.3 as Float).round(), -1.0);
        assert_biteq!((-1.5 as Float).round(), -2.0);
        assert_biteq!((-1.7 as Float).round(), -2.0);
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
        assert_biteq!((2.5 as Float).round_ties_even(), 2.0);
        assert_biteq!((1.0 as Float).round_ties_even(), 1.0);
        assert_biteq!((1.3 as Float).round_ties_even(), 1.0);
        assert_biteq!((1.5 as Float).round_ties_even(), 2.0);
        assert_biteq!((1.7 as Float).round_ties_even(), 2.0);
        assert_biteq!((0.5 as Float).round_ties_even(), 0.0);
        assert_biteq!((0.0 as Float).round_ties_even(), 0.0);
        assert_biteq!((-0.0 as Float).round_ties_even(), -0.0);
        assert_biteq!((-0.5 as Float).round_ties_even(), -0.0);
        assert_biteq!((-1.0 as Float).round_ties_even(), -1.0);
        assert_biteq!((-1.3 as Float).round_ties_even(), -1.0);
        assert_biteq!((-1.5 as Float).round_ties_even(), -2.0);
        assert_biteq!((-1.7 as Float).round_ties_even(), -2.0);
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
        assert_biteq!((1.0 as Float).trunc(), 1.0);
        assert_biteq!((1.3 as Float).trunc(), 1.0);
        assert_biteq!((1.5 as Float).trunc(), 1.0);
        assert_biteq!((1.7 as Float).trunc(), 1.0);
        assert_biteq!((0.5 as Float).trunc(), 0.0);
        assert_biteq!((0.0 as Float).trunc(), 0.0);
        assert_biteq!((-0.0 as Float).trunc(), -0.0);
        assert_biteq!((-0.5 as Float).trunc(), -0.0);
        assert_biteq!((-1.0 as Float).trunc(), -1.0);
        assert_biteq!((-1.3 as Float).trunc(), -1.0);
        assert_biteq!((-1.5 as Float).trunc(), -1.0);
        assert_biteq!((-1.7 as Float).trunc(), -1.0);
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
        assert_biteq!((1.0 as Float).fract(), 0.0);
        assert_approx_eq!((1.3 as Float).fract(), 0.3); // rounding differs between float types
        assert_biteq!((1.5 as Float).fract(), 0.5);
        assert_approx_eq!((1.7 as Float).fract(), 0.7);
        assert_biteq!((0.5 as Float).fract(), 0.5);
        assert_biteq!((0.0 as Float).fract(), 0.0);
        assert_biteq!((-0.0 as Float).fract(), 0.0);
        assert_biteq!((-0.5 as Float).fract(), -0.5);
        assert_biteq!((-1.0 as Float).fract(), 0.0);
        assert_approx_eq!((-1.3 as Float).fract(), -0.3); // rounding differs between float types
        assert_biteq!((-1.5 as Float).fract(), -0.5);
        assert_approx_eq!((-1.7 as Float).fract(), -0.7);
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

// FIXME(f16_f128,miri): many of these have to be disabled since miri does not yet support
// the intrinsics.

float_test! {
    name: sqrt_domain,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(all(not(miri), target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
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
        const: #[cfg(false)],
        f16: #[cfg(all(not(miri), target_has_reliable_f16_math))],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
    },
    test<Float> {
        use core::cmp::Ordering;

        fn quiet_bit_mask() -> <Float as TestableFloat>::Int {
            1 << (Float::MANTISSA_DIGITS - 2)
        }

        fn q_nan() -> Float {
            Float::from_bits(Float::NAN.to_bits() | quiet_bit_mask())
        }

        assert_eq!(Ordering::Equal, Float::total_cmp(&-q_nan(), &-q_nan()));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-Float::INFINITY, &-Float::INFINITY));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-Float::MAX, &-Float::MAX));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-2.5, &-2.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-1.0, &-1.0));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-1.5, &-1.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-0.5, &-0.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-Float::MIN_POSITIVE, &-Float::MIN_POSITIVE));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-Float::TINY, &-Float::TINY));
        assert_eq!(Ordering::Equal, Float::total_cmp(&-0.0, &-0.0));
        assert_eq!(Ordering::Equal, Float::total_cmp(&0.0, &0.0));
        assert_eq!(Ordering::Equal, Float::total_cmp(&Float::TINY, &Float::TINY));
        assert_eq!(Ordering::Equal, Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Equal, Float::total_cmp(&Float::MIN_POSITIVE, &Float::MIN_POSITIVE));
        assert_eq!(Ordering::Equal, Float::total_cmp(&0.5, &0.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&1.0, &1.0));
        assert_eq!(Ordering::Equal, Float::total_cmp(&1.5, &1.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&2.5, &2.5));
        assert_eq!(Ordering::Equal, Float::total_cmp(&Float::MAX, &Float::MAX));
        assert_eq!(Ordering::Equal, Float::total_cmp(&Float::INFINITY, &Float::INFINITY));
        assert_eq!(Ordering::Equal, Float::total_cmp(&q_nan(), &q_nan()));

        assert_eq!(Ordering::Less, Float::total_cmp(&-Float::INFINITY, &-Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&-Float::MAX, &-2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-2.5, &-1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-1.5, &-1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-1.0, &-0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-0.5, &-Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&-Float::MIN_POSITIVE, &-Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-Float::TINY, &-0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-0.0, &0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&0.0, &Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&Float::TINY, &Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&Float::MIN_POSITIVE, &0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&0.5, &1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&1.0, &1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&1.5, &2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&2.5, &Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&Float::MAX, &Float::INFINITY));

        assert_eq!(Ordering::Greater, Float::total_cmp(&-Float::MAX, &-Float::INFINITY));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-2.5, &-Float::MAX));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-1.5, &-2.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-1.0, &-1.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-0.5, &-1.0));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-Float::MIN_POSITIVE, &-0.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-Float::MAX_SUBNORMAL, &-Float::MIN_POSITIVE));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-Float::TINY, &-Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Greater, Float::total_cmp(&-0.0, &-Float::TINY));
        assert_eq!(Ordering::Greater, Float::total_cmp(&0.0, &-0.0));
        assert_eq!(Ordering::Greater, Float::total_cmp(&Float::TINY, &0.0));
        assert_eq!(Ordering::Greater, Float::total_cmp(&Float::MAX_SUBNORMAL, &Float::TINY));
        assert_eq!(Ordering::Greater, Float::total_cmp(&Float::MIN_POSITIVE, &Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Greater, Float::total_cmp(&0.5, &Float::MIN_POSITIVE));
        assert_eq!(Ordering::Greater, Float::total_cmp(&1.0, &0.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&1.5, &1.0));
        assert_eq!(Ordering::Greater, Float::total_cmp(&2.5, &1.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&Float::MAX, &2.5));
        assert_eq!(Ordering::Greater, Float::total_cmp(&Float::INFINITY, &Float::MAX));

        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-Float::INFINITY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &-0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &0.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &Float::TINY));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &Float::MAX_SUBNORMAL));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &Float::MIN_POSITIVE));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &0.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &1.0));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &1.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &2.5));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &Float::MAX));
        assert_eq!(Ordering::Less, Float::total_cmp(&-q_nan(), &Float::INFINITY));

    }
}

// FIXME(f16): Tests involving sNaN are disabled because without optimizations, `total_cmp` is
// getting incorrectly lowered to code that includes a `extend`/`trunc` round trip, which quiets
// sNaNs. See: https://github.com/llvm/llvm-project/issues/104915

float_test! {
    name: total_cmp_s_nan,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(false)],
        f128: #[cfg(all(not(miri), target_has_reliable_f128_math))],
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
