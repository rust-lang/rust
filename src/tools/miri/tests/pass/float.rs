#![feature(stmt_expr_attributes)]
#![feature(float_erf)]
#![feature(float_gamma)]
#![feature(core_intrinsics)]
#![feature(f128)]
#![feature(f16)]
#![allow(arithmetic_overflow)]
#![allow(internal_features)]
#![allow(unnecessary_transmutes)]

use std::any::type_name;
use std::cmp::min;
use std::fmt::{Debug, Display, LowerHex};
use std::hint::black_box;
use std::{f32, f64};

/// Compare the two floats, allowing for $ulp many ULPs of error.
///
/// ULP means "Units in the Last Place" or "Units of Least Precision".
/// The ULP of a float `a`` is the smallest possible change at `a`, so the ULP difference represents how
/// many discrete floating-point steps are needed to reach the actual value from the expected value.
///
/// Essentially ULP can be seen as a distance metric of floating-point numbers, but with
/// the same amount of "spacing" between all consecutive representable values. So even though 2 very large floating point numbers
/// have a large value difference, their ULP can still be 1, so they are still "approximatly equal",
/// but the EPSILON check would have failed.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $ulp:expr) => {{
        let (actual, expected) = ($a, $b);
        let allowed_ulp_diff = $ulp;
        let _force_same_type = actual == expected;
        // Approximate the ULP by taking half the distance between the number one place "up"
        // and the number one place "down".
        let ulp = (expected.next_up() - expected.next_down()) / 2.0;
        let ulp_diff = ((actual - expected) / ulp).abs().round() as i32;
        if ulp_diff > allowed_ulp_diff {
            panic!("{actual:?} is not approximately equal to {expected:?}\ndifference in ULP: {ulp_diff} > {allowed_ulp_diff}");
        };
    }};

    ($a:expr, $b: expr) => {
        // accept up to 12ULP (4ULP for host floats and 4ULP for miri artificial error and 4 for any additional effects
        // due to having multiple error sources.
        assert_approx_eq!($a, $b, 12);
    };
}


/// From IEEE 754 a Signaling NaN for single precision has the following representation:
/// ```
/// s | 1111 1111 | 0x..x
/// ````
/// Were at least one `x` is a 1.
///
/// This sNaN has the following representation and is used for testing purposes.:
/// ```
/// 0 | 1111111 | 01..0
/// ```
const SNAN_F32: f32 = f32::from_bits(0x7fa00000);

/// From IEEE 754 a Signaling NaN for double precision has the following representation:
/// ```
/// s | 1111 1111 111 | 0x..x
/// ````
/// Were at least one `x` is a 1.
///
/// This sNaN has the following representation and is used for testing purposes.:
/// ```
/// 0 | 1111 1111 111 | 01..0
/// ```
const SNAN_F64: f64 = f64::from_bits(0x7ff4000000000000);

fn main() {
    basic();
    casts();
    more_casts();
    ops();
    nan_casts();
    rounding();
    mul_add();
    libm();
    test_fast();
    test_algebraic();
    test_fmuladd();
    test_min_max_nondet();
    test_non_determinism();
}

trait Float: Copy + PartialEq + Debug {
    /// The unsigned integer with the same bit width as this float
    type Int: Copy + PartialEq + LowerHex + Debug;
    const BITS: u32 = size_of::<Self>() as u32 * 8;
    const EXPONENT_BITS: u32 = Self::BITS - Self::SIGNIFICAND_BITS - 1;
    const SIGNIFICAND_BITS: u32;

    /// The saturated (all ones) value of the exponent (infinity representation)
    const EXPONENT_SAT: u32 = (1 << Self::EXPONENT_BITS) - 1;

    /// The exponent bias value (max representable positive exponent)
    const EXPONENT_BIAS: u32 = Self::EXPONENT_SAT >> 1;

    fn to_bits(self) -> Self::Int;
}

macro_rules! impl_float {
    ($ty:ty, $ity:ty) => {
        impl Float for $ty {
            type Int = $ity;
            // Just get this from std's value, which includes the implicit digit
            const SIGNIFICAND_BITS: u32 = <$ty>::MANTISSA_DIGITS - 1;

            fn to_bits(self) -> Self::Int {
                self.to_bits()
            }
        }
    };
}

impl_float!(f16, u16);
impl_float!(f32, u32);
impl_float!(f64, u64);
impl_float!(f128, u128);

trait FloatToInt<Int>: Copy {
    fn cast(self) -> Int;
    unsafe fn cast_unchecked(self) -> Int;
}

macro_rules! float_to_int {
    ($fty:ty => $($ity:ty),+ $(,)?) => {
        $(
            impl FloatToInt<$ity> for $fty {
                fn cast(self) -> $ity {
                    self as _
                }
                unsafe fn cast_unchecked(self) -> $ity {
                    self.to_int_unchecked()
                }
            }
        )*
    };
}

float_to_int!(f16 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
float_to_int!(f32 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
float_to_int!(f64 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
float_to_int!(f128 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

/// Test this cast both via `as` and via `approx_unchecked` (i.e., it must not saturate).
#[track_caller]
#[inline(never)]
fn test_both_cast<F, I>(x: F, y: I, msg: impl Display)
where
    F: FloatToInt<I>,
    I: PartialEq + Debug,
{
    let f_tname = type_name::<F>();
    let i_tname = type_name::<I>();
    assert_eq!(x.cast(), y, "{f_tname} -> {i_tname}: {msg}");
    assert_eq!(unsafe { x.cast_unchecked() }, y, "{f_tname} -> {i_tname}: {msg}",);
}

/// Helper function to avoid promotion so that this tests "run-time" casts, not CTFE.
/// Doesn't make a big difference when running this in Miri, but it means we can compare this
/// with the LLVM backend by running `rustc -Zmir-opt-level=0 -Zsaturating-float-casts`.
#[track_caller]
#[inline(never)]
fn assert_eq<T: PartialEq + Debug>(x: T, y: T) {
    assert_eq!(x, y);
}

/// The same as `assert_eq` except prints a specific message on failure
#[track_caller]
#[inline(never)]
fn assert_eq_msg<T: PartialEq + Debug>(x: T, y: T, msg: impl Display) {
    assert_eq!(x, y, "{msg}");
}

/// Check that floats have bitwise equality
fn assert_biteq<F: Float>(a: F, b: F, msg: impl Display) {
    let ab = a.to_bits();
    let bb = b.to_bits();
    let tname = type_name::<F>();
    let width = (2 + F::BITS / 4) as usize;
    assert_eq_msg::<F::Int>(
        ab,
        bb,
        format_args!("({ab:#0width$x} != {bb:#0width$x}) {tname}: {msg}"),
    );
}

/// Check that two floats have equality
fn assert_feq<F: Float>(a: F, b: F, msg: impl Display) {
    let ab = a.to_bits();
    let bb = b.to_bits();
    let tname = type_name::<F>();
    let width = (2 + F::BITS / 4) as usize;
    assert_eq_msg::<F>(a, b, format_args!("({ab:#0width$x} != {bb:#0width$x}) {tname}: {msg}"));
}

fn basic() {
    // basic arithmetic
    assert_eq(6.0_f16 * 6.0_f16, 36.0_f16);
    assert_eq(6.0_f32 * 6.0_f32, 36.0_f32);
    assert_eq(6.0_f64 * 6.0_f64, 36.0_f64);
    assert_eq(6.0_f128 * 6.0_f128, 36.0_f128);
    assert_eq(-{ 5.0_f16 }, -5.0_f16);
    assert_eq(-{ 5.0_f32 }, -5.0_f32);
    assert_eq(-{ 5.0_f64 }, -5.0_f64);
    assert_eq(-{ 5.0_f128 }, -5.0_f128);

    // infinities, NaN
    assert!((5.0_f16 / 0.0).is_infinite());
    assert_ne!({ 5.0_f16 / 0.0 }, { -5.0_f16 / 0.0 });
    assert!((5.0_f32 / 0.0).is_infinite());
    assert_ne!({ 5.0_f32 / 0.0 }, { -5.0_f32 / 0.0 });
    assert!((5.0_f64 / 0.0).is_infinite());
    assert_ne!({ 5.0_f64 / 0.0 }, { 5.0_f64 / -0.0 });
    assert!((5.0_f128 / 0.0).is_infinite());
    assert_ne!({ 5.0_f128 / 0.0 }, { 5.0_f128 / -0.0 });
    assert_ne!(f16::NAN, f16::NAN);
    assert_ne!(f32::NAN, f32::NAN);
    assert_ne!(f64::NAN, f64::NAN);
    assert_ne!(f128::NAN, f128::NAN);

    // negative zero
    let posz = 0.0f16;
    let negz = -0.0f16;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());
    let posz = 0.0f32;
    let negz = -0.0f32;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());
    let posz = 0.0f64;
    let negz = -0.0f64;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());
    let posz = 0.0f128;
    let negz = -0.0f128;
    assert_eq(posz, negz);
    assert_ne!(posz.to_bits(), negz.to_bits());

    // byte-level transmute
    let x: u16 = unsafe { std::mem::transmute(42.0_f16) };
    let y: f16 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f16);
    let x: u32 = unsafe { std::mem::transmute(42.0_f32) };
    let y: f32 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f32);
    let x: u64 = unsafe { std::mem::transmute(42.0_f64) };
    let y: f64 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f64);
    let x: u128 = unsafe { std::mem::transmute(42.0_f128) };
    let y: f128 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f128);

    // `%` sign behavior, some of this used to be buggy
    assert!((black_box(1.0f16) % 1.0).is_sign_positive());
    assert!((black_box(1.0f16) % -1.0).is_sign_positive());
    assert!((black_box(-1.0f16) % 1.0).is_sign_negative());
    assert!((black_box(-1.0f16) % -1.0).is_sign_negative());
    assert!((black_box(1.0f32) % 1.0).is_sign_positive());
    assert!((black_box(1.0f32) % -1.0).is_sign_positive());
    assert!((black_box(-1.0f32) % 1.0).is_sign_negative());
    assert!((black_box(-1.0f32) % -1.0).is_sign_negative());
    assert!((black_box(1.0f64) % 1.0).is_sign_positive());
    assert!((black_box(1.0f64) % -1.0).is_sign_positive());
    assert!((black_box(-1.0f64) % 1.0).is_sign_negative());
    assert!((black_box(-1.0f64) % -1.0).is_sign_negative());
    assert!((black_box(1.0f128) % 1.0).is_sign_positive());
    assert!((black_box(1.0f128) % -1.0).is_sign_positive());
    assert!((black_box(-1.0f128) % 1.0).is_sign_negative());
    assert!((black_box(-1.0f128) % -1.0).is_sign_negative());

    assert_eq!((-1.0f16).abs(), 1.0f16);
    assert_eq!(34.2f16.abs(), 34.2f16);
    assert_eq!((-1.0f32).abs(), 1.0f32);
    assert_eq!(34.2f32.abs(), 34.2f32);
    assert_eq!((-1.0f64).abs(), 1.0f64);
    assert_eq!(34.2f64.abs(), 34.2f64);
    assert_eq!((-1.0f128).abs(), 1.0f128);
    assert_eq!(34.2f128.abs(), 34.2f128);
}

/// Test casts from floats to ints and back
macro_rules! test_ftoi_itof {
    (
        f: $fty:ty,
        i: $ity:ty,
        // Int min and max as float literals
        imin_f: $imin_f:literal,
        imax_f: $imax_f:literal $(,)?
    ) => {{
        /// By default we test float to int `as` casting as well as to_int_unchecked
        fn assert_ftoi(f: $fty, i: $ity, msg: &str) {
            #[allow(unused_comparisons)]
            if <$ity>::MIN >= 0 && f < 0.0 {
                // If `ity` is signed and `f` is negative, it is unrepresentable so skip
                // unchecked casts.
                assert_ftoi_unrep(f, i, msg);
            } else {
                test_both_cast::<$fty, $ity>(f, i, msg);
            }
        }

        /// Unrepresentable values only get tested with `as` casting, not unchecked
        fn assert_ftoi_unrep(f: $fty, i: $ity, msg: &str) {
            assert_eq_msg::<$ity>(
                f as $ity,
                i,
                format_args!("{} -> {}: {msg}", stringify!($fty), stringify!($ity)),
            );
        }

        /// Int to float checks
        fn assert_itof(i: $ity, f: $fty, msg: &str) {
            assert_eq_msg::<$fty>(
                i as $fty,
                f,
                format_args!("{} -> {}: {msg}", stringify!($ity), stringify!($fty)),
            );
        }

        /// Check both float to int and int to float
        fn assert_bidir(f: $fty, i: $ity, msg: &str) {
            assert_ftoi(f, i, msg);
            assert_itof(i, f, msg);
        }

        /// Check both float to int and int to float for unrepresentable numbers
        fn assert_bidir_unrep(f: $fty, i: $ity, msg: &str) {
            assert_ftoi_unrep(f, i, msg);
            assert_itof(i, f, msg);
        }

        let fbits = <$fty>::BITS;
        let fsig_bits = <$fty>::SIGNIFICAND_BITS;
        let ibits = <$ity>::BITS;
        let imax: $ity = <$ity>::MAX;
        let imin: $ity = <$ity>::MIN;
        let izero: $ity = 0;
        #[allow(unused_comparisons)]
        let isigned = <$ity>::MIN < 0;

        #[allow(overflowing_literals)]
        let imin_f: $fty = $imin_f;
        #[allow(overflowing_literals)]
        let imax_f: $fty = $imax_f;

        // If an integer can fit entirely in the mantissa (counting the hidden bit), every value
        // can be represented exactly.
        let all_ints_exact_rep = ibits <= fsig_bits + 1;

        // We can represent the full range of the integer (but possibly not every value) without
        // saturating to infinity if `1 << (I::BITS - 1)` (single one in the MSB position) is
        // within the float's dynamic range.
        let int_range_rep = ibits - 1 < <$fty>::EXPONENT_BIAS;

        // Skip unchecked cast when int min/max would be unrepresentable
        let assert_ftoi_big = if all_ints_exact_rep { assert_ftoi } else { assert_ftoi_unrep };
        let assert_bidir_big = if all_ints_exact_rep { assert_bidir } else { assert_bidir_unrep };

        // Near zero representations
        assert_bidir(0.0, 0, "zero");
        assert_ftoi(-0.0, 0, "negative zero");
        assert_ftoi(1.0, 1, "one");
        assert_ftoi(-1.0, izero.saturating_sub(1), "negative one");
        assert_ftoi(1.0 - <$fty>::EPSILON, 0, "1.0 - ε");
        assert_ftoi(1.0 + <$fty>::EPSILON, 1, "1.0 + ε");
        assert_ftoi(-1.0 + <$fty>::EPSILON, 0, "-1.0 + ε");
        assert_ftoi(-1.0 - <$fty>::EPSILON, izero.saturating_sub(1), "-1.0 - ε");
        assert_ftoi(<$fty>::from_bits(0x1), 0, "min subnormal");
        assert_ftoi(<$fty>::from_bits(0x1 | 1 << (fbits - 1)), 0, "min neg subnormal");

        // Spot checks. Use `saturating_sub` to create negative integers so that unsigned
        // integers stay at zero.
        assert_ftoi(0.9, 0, "0.9");
        assert_ftoi(-0.9, 0, "-0.9");
        assert_ftoi(1.1, 1, "1.1");
        assert_ftoi(-1.1, izero.saturating_sub(1), "-1.1");
        assert_ftoi(1.9, 1, "1.9");
        assert_ftoi(-1.9, izero.saturating_sub(1), "-1.9");
        assert_ftoi(5.0, 5, "5.0");
        assert_ftoi(-5.0, izero.saturating_sub(5), "-5.0");
        assert_ftoi(5.9, 5, "5.0");
        assert_ftoi(-5.9, izero.saturating_sub(5), "-5.0");

        // Exercise the middle of the integer's bit range. A power of two fits as long as the
        // exponent can fit its log2, so cap at the maximum representable power of two (which
        // is the exponent's bias).
        let half_i_max: $ity = 1 << min(ibits / 2, <$fty>::EXPONENT_BIAS);
        let half_i_min = izero.saturating_sub(half_i_max);
        assert_bidir(half_i_max as $fty, half_i_max, "half int max");
        assert_bidir(half_i_min as $fty, half_i_min, "half int min");

        // Integer limits
        assert_bidir_big(imax_f, imax, "i max");
        assert_bidir_big(imin_f, imin, "i min");

        // We need a small perturbation to test against that does not round up to the next
        // integer. `f16` needs a smaller perturbation since it only has resolution for ~1 decimal
        // place near 10^3.
        let perturb = if fbits < 32 { 0.9 } else { 0.99 };
        assert_ftoi_big(imax_f + perturb, <$ity>::MAX, "slightly above i max");
        assert_ftoi_big(imin_f - perturb, <$ity>::MIN, "slightly below i min");

        // Tests for when we can represent the integer's magnitude
        if int_range_rep {
            // If the float can represent values larger than the integer, float extremes
            // will saturate.
            assert_ftoi_unrep(<$fty>::MAX, imax, "f max");
            assert_ftoi_unrep(<$fty>::MIN, imin, "f min");

            // Max representable power of 10
            let pow10_max = (10 as $ity).pow(imax.ilog10());

            // If the power of 10 should be representable (fits in a mantissa), check it
            if ibits - pow10_max.leading_zeros() - pow10_max.trailing_zeros() <= fsig_bits + 1 {
                assert_bidir(pow10_max as $fty, pow10_max, "pow10 max");
            }
        }

        // Test rounding the largest and smallest integers, but skip this when
        // all integers have an exact representation (it's less interesting then and the arithmetic gets more complicated).
        if int_range_rep && !all_ints_exact_rep {
            // The maximum representable integer is a saturated mantissa (including the implicit
            // bit), shifted into the int's leftmost position.
            //
            // Positive signed integers never use their top bit, so shift by one bit fewer.
            let sat_mantissa: $ity = (1 << (fsig_bits + 1)) - 1;
            let adj = if isigned { 1 } else { 0 };
            let max_rep = sat_mantissa << (sat_mantissa.leading_zeros() - adj);

            // This value should roundtrip exactly
            assert_bidir(max_rep as $fty, max_rep, "max representable int");

            // The cutoff for where to round to `imax` is halfway between the maximum exactly
            // representable integer and `imax`. This should round down (to `max_rep`),
            // i.e., `max_rep as $fty == max_non_sat as $fty`.
            let max_non_sat = max_rep + ((imax - max_rep) / 2);
            assert_bidir(max_non_sat as $fty, max_rep, "max non saturating int");

            // So the next value up should round up to the maximum value of the integer
            assert_bidir_unrep((max_non_sat + 1) as $fty, imax, "min infinite int");

            if isigned {
                // Floats can always represent the minimum signed number if they can fit the
                // exponent, because it is just a `1` in the MSB. So, no negative int -> float
                // conversion will round to negative infinity (if the exponent fits).
                //
                // Since `imin` is thus the minimum representable value, we test rounding near
                // the next value. This happens to be the opposite of the maximum representable
                // value, and it should roundtrip exactly.
                let next_min_rep = max_rep.wrapping_neg();
                assert_bidir(next_min_rep as $fty, next_min_rep, "min representable above imin");

                // Following a similar pattern as for positive numbers, halfway between this value
                // and `imin` should round back to `next_min_rep`.
                let min_non_sat = imin - ((imin - next_min_rep) / 2) + 1;
                assert_bidir(
                    min_non_sat as $fty,
                    next_min_rep,
                    "min int that does not round to imin",
                );

                // And then anything else saturates to the minimum value.
                assert_bidir_unrep(
                    (min_non_sat - 1) as $fty,
                    imin,
                    "max negative int that rounds to imin",
                );
            }
        }

        // Check potentially saturating int ranges. (`imax_f` here will be `$fty::INFINITY` if
        // it cannot be represented as a finite value.)
        assert_itof(imax, imax_f, "imax");
        assert_itof(imin, imin_f, "imin");

        // Float limits
        assert_ftoi_unrep(<$fty>::INFINITY, imax, "f inf");
        assert_ftoi_unrep(<$fty>::NEG_INFINITY, imin, "f neg inf");
        assert_ftoi_unrep(<$fty>::NAN, 0, "f nan");
        assert_ftoi_unrep(-<$fty>::NAN, 0, "f neg nan");
    }};
}

/// Test casts from one float to another
macro_rules! test_ftof {
    (
        f1: $f1:ty,
        f2: $f2:ty $(,)?
    ) => {{
        type F2Int = <$f2 as Float>::Int;

        let f1zero: $f1 = 0.0;
        let f2zero: $f2 = 0.0;
        let f1five: $f1 = 5.0;
        let f2five: $f2 = 5.0;

        assert_biteq((f1zero as $f2), f2zero, "0.0");
        assert_biteq(((-f1zero) as $f2), (-f2zero), "-0.0");
        assert_biteq((f1five as $f2), f2five, "5.0");
        assert_biteq(((-f1five) as $f2), (-f2five), "-5.0");

        assert_feq(<$f1>::INFINITY as $f2, <$f2>::INFINITY, "max -> inf");
        assert_feq(<$f1>::NEG_INFINITY as $f2, <$f2>::NEG_INFINITY, "max -> inf");
        assert!((<$f1>::NAN as $f2).is_nan(), "{} -> {} nan", stringify!($f1), stringify!($f2));

        let min_sub_casted = <$f1>::from_bits(0x1) as $f2;
        let min_neg_sub_casted = <$f1>::from_bits(0x1 | 1 << (<$f1>::BITS - 1)) as $f2;

        if <$f1>::BITS > <$f2>::BITS {
            assert_feq(<$f1>::MAX as $f2, <$f2>::INFINITY, "max -> inf");
            assert_feq(<$f1>::MIN as $f2, <$f2>::NEG_INFINITY, "max -> inf");
            assert_biteq(min_sub_casted, f2zero, "min subnormal -> 0.0");
            assert_biteq(min_neg_sub_casted, -f2zero, "min neg subnormal -> -0.0");
        } else {
            // When increasing precision, the minimum subnormal will just roll to the next
            // exponent. This exponent will be the current exponent (with bias), plus
            // `sig_bits - 1` to account for the implicit change in exponent (since the
            // mantissa starts with 0).
            let sub_casted = <$f2>::from_bits(
                ((<$f2>::EXPONENT_BIAS - (<$f1>::EXPONENT_BIAS + <$f1>::SIGNIFICAND_BITS - 1))
                    as F2Int)
                    << <$f2>::SIGNIFICAND_BITS,
            );
            assert_biteq(min_sub_casted, sub_casted, "min subnormal");
            assert_biteq(min_neg_sub_casted, -sub_casted, "min neg subnormal");
        }
    }};
}

/// Many of these test patterns were adapted from the values in
/// https://github.com/WebAssembly/testsuite/blob/master/conversions.wast.
fn casts() {
    /* int <-> float generic tests */

    test_ftoi_itof! { f: f16, i: i8, imin_f: -128.0, imax_f: 127.0 };
    test_ftoi_itof! { f: f16, i: u8, imin_f: 0.0, imax_f: 255.0 };
    test_ftoi_itof! { f: f16, i: i16, imin_f: -32_768.0, imax_f: 32_767.0 };
    test_ftoi_itof! { f: f16, i: u16, imin_f: 0.0, imax_f: 65_535.0 };
    test_ftoi_itof! { f: f16, i: i32, imin_f: -2_147_483_648.0, imax_f: 2_147_483_647.0 };
    test_ftoi_itof! { f: f16, i: u32, imin_f: 0.0, imax_f: 4_294_967_295.0 };
    test_ftoi_itof! {
        f: f16,
        i: i64,
        imin_f: -9_223_372_036_854_775_808.0,
        imax_f: 9_223_372_036_854_775_807.0
    };
    test_ftoi_itof! { f: f16, i: u64, imin_f: 0.0, imax_f: 18_446_744_073_709_551_615.0 };
    test_ftoi_itof! {
        f: f16,
        i: i128,
        imin_f: -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
        imax_f: 170_141_183_460_469_231_731_687_303_715_884_105_727.0,
    };
    test_ftoi_itof! {
        f: f16,
        i: u128,
        imin_f: 0.0,
        imax_f: 340_282_366_920_938_463_463_374_607_431_768_211_455.0
    };

    test_ftoi_itof! { f: f32, i: i8, imin_f: -128.0, imax_f: 127.0 };
    test_ftoi_itof! { f: f32, i: u8, imin_f: 0.0, imax_f: 255.0 };
    test_ftoi_itof! { f: f32, i: i16, imin_f: -32_768.0, imax_f: 32_767.0 };
    test_ftoi_itof! { f: f32, i: u16, imin_f: 0.0, imax_f: 65_535.0 };
    test_ftoi_itof! { f: f32, i: i32, imin_f: -2_147_483_648.0, imax_f: 2_147_483_647.0 };
    test_ftoi_itof! { f: f32, i: u32, imin_f: 0.0, imax_f: 4_294_967_295.0 };
    test_ftoi_itof! {
        f: f32,
        i: i64,
        imin_f: -9_223_372_036_854_775_808.0,
        imax_f: 9_223_372_036_854_775_807.0
    };
    test_ftoi_itof! { f: f32, i: u64, imin_f: 0.0, imax_f: 18_446_744_073_709_551_615.0 };
    test_ftoi_itof! {
        f: f32,
        i: i128,
        imin_f: -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
        imax_f: 170_141_183_460_469_231_731_687_303_715_884_105_727.0,
    };
    test_ftoi_itof! {
        f: f32,
        i: u128,
        imin_f: 0.0,
        imax_f: 340_282_366_920_938_463_463_374_607_431_768_211_455.0
    };

    test_ftoi_itof! { f: f64, i: i8, imin_f: -128.0, imax_f: 127.0 };
    test_ftoi_itof! { f: f64, i: u8, imin_f: 0.0, imax_f: 255.0 };
    test_ftoi_itof! { f: f64, i: i16, imin_f: -32_768.0, imax_f: 32_767.0 };
    test_ftoi_itof! { f: f64, i: u16, imin_f: 0.0, imax_f: 65_535.0 };
    test_ftoi_itof! { f: f64, i: i32, imin_f: -2_147_483_648.0, imax_f: 2_147_483_647.0 };
    test_ftoi_itof! { f: f64, i: u32, imin_f: 0.0, imax_f: 4_294_967_295.0 };
    test_ftoi_itof! {
        f: f64,
        i: i64,
        imin_f: -9_223_372_036_854_775_808.0,
        imax_f: 9_223_372_036_854_775_807.0
    };
    test_ftoi_itof! { f: f64, i: u64, imin_f: 0.0, imax_f: 18_446_744_073_709_551_615.0 };
    test_ftoi_itof! {
        f: f64,
        i: i128,
        imin_f: -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
        imax_f: 170_141_183_460_469_231_731_687_303_715_884_105_727.0,
    };
    test_ftoi_itof! {
        f: f64,
        i: u128,
        imin_f: 0.0,
        imax_f: 340_282_366_920_938_463_463_374_607_431_768_211_455.0
    };

    test_ftoi_itof! { f: f128, i: i8, imin_f: -128.0, imax_f: 127.0 };
    test_ftoi_itof! { f: f128, i: u8, imin_f: 0.0, imax_f: 255.0 };
    test_ftoi_itof! { f: f128, i: i16, imin_f: -32_768.0, imax_f: 32_767.0 };
    test_ftoi_itof! { f: f128, i: u16, imin_f: 0.0, imax_f: 65_535.0 };
    test_ftoi_itof! { f: f128, i: i32, imin_f: -2_147_483_648.0, imax_f: 2_147_483_647.0 };
    test_ftoi_itof! { f: f128, i: u32, imin_f: 0.0, imax_f: 4_294_967_295.0 };
    test_ftoi_itof! {
        f: f128,
        i: i64,
        imin_f: -9_223_372_036_854_775_808.0,
        imax_f: 9_223_372_036_854_775_807.0
    };
    test_ftoi_itof! { f: f128, i: u64, imin_f: 0.0, imax_f: 18_446_744_073_709_551_615.0 };
    test_ftoi_itof! {
        f: f128,
        i: i128,
        imin_f: -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
        imax_f: 170_141_183_460_469_231_731_687_303_715_884_105_727.0,
    };
    test_ftoi_itof! {
        f: f128,
        i: u128,
        imin_f: 0.0,
        imax_f: 340_282_366_920_938_463_463_374_607_431_768_211_455.0
    };

    /* int <-> float spot checks */

    // int -> f32
    assert_eq::<f32>(1234567890i32 as f32, /*0x1.26580cp+30*/ f32::from_bits(0x4e932c06));
    assert_eq::<f32>(
        0x7fffff4000000001i64 as f32,
        /*0x1.fffffep+62*/ f32::from_bits(0x5effffff),
    );
    assert_eq::<f32>(
        0x8000004000000001u64 as i64 as f32,
        /*-0x1.fffffep+62*/ f32::from_bits(0xdeffffff),
    );
    assert_eq::<f32>(
        0x0020000020000001i64 as f32,
        /*0x1.000002p+53*/ f32::from_bits(0x5a000001),
    );
    assert_eq::<f32>(
        0xffdfffffdfffffffu64 as i64 as f32,
        /*-0x1.000002p+53*/ f32::from_bits(0xda000001),
    );

    // int -> f64
    assert_eq::<f64>(987654321i32 as f64, 987654321.0);
    assert_eq::<f64>(4669201609102990i64 as f64, 4669201609102990.0); // Feigenbaum (?)
    assert_eq::<f64>(9007199254740993i64 as f64, 9007199254740992.0);
    assert_eq::<f64>(-9007199254740993i64 as f64, -9007199254740992.0);
    assert_eq::<f64>(9007199254740995i64 as f64, 9007199254740996.0);
    assert_eq::<f64>(-9007199254740995i64 as f64, -9007199254740996.0);

    /* float -> float generic tests */

    test_ftof! { f1: f16, f2: f32 };
    test_ftof! { f1: f16, f2: f64 };
    test_ftof! { f1: f16, f2: f128 };
    test_ftof! { f1: f32, f2: f16 };
    test_ftof! { f1: f32, f2: f64 };
    test_ftof! { f1: f32, f2: f128 };
    test_ftof! { f1: f64, f2: f16 };
    test_ftof! { f1: f64, f2: f32 };
    test_ftof! { f1: f64, f2: f128 };
    test_ftof! { f1: f128, f2: f16 };
    test_ftof! { f1: f128, f2: f32 };
    test_ftof! { f1: f128, f2: f64 };

    /* float -> float spot checks */

    // f32 -> f64
    assert_eq::<f64>(
        /*0x1.fffffep+127*/ f32::from_bits(0x7f7fffff) as f64,
        /*0x1.fffffep+127*/ f64::from_bits(0x47efffffe0000000),
    );
    assert_eq::<f64>(
        /*-0x1.fffffep+127*/ (-f32::from_bits(0x7f7fffff)) as f64,
        /*-0x1.fffffep+127*/ -f64::from_bits(0x47efffffe0000000),
    );
    assert_eq::<f64>(
        /*0x1p-119*/ f32::from_bits(0x4000000) as f64,
        /*0x1p-119*/ f64::from_bits(0x3880000000000000),
    );
    assert_eq::<f64>(
        /*0x1.8f867ep+125*/ f32::from_bits(0x7e47c33f) as f64,
        6.6382536710104395e+37,
    );

    // f64 -> f32
    assert_eq::<f32>(
        /*0x1.fffffe0000000p-127*/ f64::from_bits(0x380fffffe0000000) as f32,
        /*0x1p-149*/ f32::from_bits(0x800000),
    );
    assert_eq::<f32>(
        /*0x1.4eae4f7024c7p+108*/ f64::from_bits(0x46b4eae4f7024c70) as f32,
        /*0x1.4eae5p+108*/ f32::from_bits(0x75a75728),
    );
}

fn ops() {
    // f16 min/max
    assert_eq((1.0_f16).max(-1.0), 1.0);
    assert_eq((1.0_f16).min(-1.0), -1.0);
    assert_eq(f16::NAN.min(9.0), 9.0);
    assert_eq(f16::NAN.max(-9.0), -9.0);
    assert_eq((9.0_f16).min(f16::NAN), 9.0);
    assert_eq((-9.0_f16).max(f16::NAN), -9.0);

    // f32 min/max
    assert_eq((1.0 as f32).max(-1.0), 1.0);
    assert_eq((1.0 as f32).min(-1.0), -1.0);
    assert_eq(f32::NAN.min(9.0), 9.0);
    assert_eq(f32::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f32).min(f32::NAN), 9.0);
    assert_eq((-9.0 as f32).max(f32::NAN), -9.0);

    // f64 min/max
    assert_eq((1.0 as f64).max(-1.0), 1.0);
    assert_eq((1.0 as f64).min(-1.0), -1.0);
    assert_eq(f64::NAN.min(9.0), 9.0);
    assert_eq(f64::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f64).min(f64::NAN), 9.0);
    assert_eq((-9.0 as f64).max(f64::NAN), -9.0);

    // f128 min/max
    assert_eq((1.0_f128).max(-1.0), 1.0);
    assert_eq((1.0_f128).min(-1.0), -1.0);
    assert_eq(f128::NAN.min(9.0), 9.0);
    assert_eq(f128::NAN.max(-9.0), -9.0);
    assert_eq((9.0_f128).min(f128::NAN), 9.0);
    assert_eq((-9.0_f128).max(f128::NAN), -9.0);

    // f16 copysign
    assert_eq(3.5_f16.copysign(0.42), 3.5_f16);
    assert_eq(3.5_f16.copysign(-0.42), -3.5_f16);
    assert_eq((-3.5_f16).copysign(0.42), 3.5_f16);
    assert_eq((-3.5_f16).copysign(-0.42), -3.5_f16);
    assert!(f16::NAN.copysign(1.0).is_nan());

    // f32 copysign
    assert_eq(3.5_f32.copysign(0.42), 3.5_f32);
    assert_eq(3.5_f32.copysign(-0.42), -3.5_f32);
    assert_eq((-3.5_f32).copysign(0.42), 3.5_f32);
    assert_eq((-3.5_f32).copysign(-0.42), -3.5_f32);
    assert!(f32::NAN.copysign(1.0).is_nan());

    // f64 copysign
    assert_eq(3.5_f64.copysign(0.42), 3.5_f64);
    assert_eq(3.5_f64.copysign(-0.42), -3.5_f64);
    assert_eq((-3.5_f64).copysign(0.42), 3.5_f64);
    assert_eq((-3.5_f64).copysign(-0.42), -3.5_f64);
    assert!(f64::NAN.copysign(1.0).is_nan());

    // f128 copysign
    assert_eq(3.5_f128.copysign(0.42), 3.5_f128);
    assert_eq(3.5_f128.copysign(-0.42), -3.5_f128);
    assert_eq((-3.5_f128).copysign(0.42), 3.5_f128);
    assert_eq((-3.5_f128).copysign(-0.42), -3.5_f128);
    assert!(f128::NAN.copysign(1.0).is_nan());
}

/// Tests taken from rustc test suite.
///

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => (
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run-time {} -> {}", stringify!($src_ty), stringify!($dest_ty));

        {
            const X: $src_ty = $val;
            const Y: $dest_ty = X as $dest_ty;
            assert_eq!(Y, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
    );

    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test!($fval, f32 -> $ity, $ival);
        test!($fval, f64 -> $ity, $ival);
    )
}

macro_rules! common_fptoi_tests {
    ($fty:ident -> $($ity:ident)+) => ({ $(
        test!($fty::NAN, $fty -> $ity, 0);
        test!($fty::INFINITY, $fty -> $ity, $ity::MAX);
        test!($fty::NEG_INFINITY, $fty -> $ity, $ity::MIN);
        // These two tests are not solely float->int tests, in particular the latter relies on
        // `u128::MAX as f32` not being UB. But that's okay, since this file tests int->float
        // as well, the test is just slightly misplaced.
        test!($ity::MIN as $fty, $fty -> $ity, $ity::MIN);
        test!($ity::MAX as $fty, $fty -> $ity, $ity::MAX);
        test!(0., $fty -> $ity, 0);
        test!($fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.9, $fty -> $ity, 0);
        test!(1., $fty -> $ity, 1);
        test!(42., $fty -> $ity, 42);
    )+ });

    (f* -> $($ity:ident)+) => ({
        common_fptoi_tests!(f32 -> $($ity)+);
        common_fptoi_tests!(f64 -> $($ity)+);
    })
}

macro_rules! fptoui_tests {
    ($fty: ident -> $($ity: ident)+) => ({ $(
        test!(-0., $fty -> $ity, 0);
        test!(-$fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.99999994, $fty -> $ity, 0);
        test!(-1., $fty -> $ity, 0);
        test!(-100., $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e50, $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e130, $fty -> $ity, 0);
    )+ });

    (f* -> $($ity:ident)+) => ({
        fptoui_tests!(f32 -> $($ity)+);
        fptoui_tests!(f64 -> $($ity)+);
    })
}

fn more_casts() {
    common_fptoi_tests!(f* -> i8 i16 i32 i64 u8 u16 u32 u64);
    fptoui_tests!(f* -> u8 u16 u32 u64);
    common_fptoi_tests!(f* -> i128 u128);
    fptoui_tests!(f* -> u128);

    // The following tests cover edge cases for some integer types.

    // # u8
    test!(254., f* -> u8, 254);
    test!(256., f* -> u8, 255);

    // # i8
    test!(-127., f* -> i8, -127);
    test!(-129., f* -> i8, -128);
    test!(126., f* -> i8, 126);
    test!(128., f* -> i8, 127);

    // # i32
    // -2147483648. is i32::MIN (exactly)
    test!(-2147483648., f* -> i32, i32::MIN);
    // 2147483648. is i32::MAX rounded up
    test!(2147483648., f32 -> i32, 2147483647);
    // With 24 significand bits, floats with magnitude in [2^30 + 1, 2^31] are rounded to
    // multiples of 2^7. Therefore, nextDown(round(i32::MAX)) is 2^31 - 128:
    test!(2147483520., f32 -> i32, 2147483520);
    // Similarly, nextUp(i32::MIN) is i32::MIN + 2^8 and nextDown(i32::MIN) is i32::MIN - 2^7
    test!(-2147483904., f* -> i32, i32::MIN);
    test!(-2147483520., f* -> i32, -2147483520);

    // # u32
    // round(MAX) and nextUp(round(MAX))
    test!(4294967040., f* -> u32, 4294967040);
    test!(4294967296., f* -> u32, 4294967295);

    // # u128
    // float->int:
    test!(f32::MAX, f32 -> u128, 0xffffff00000000000000000000000000);
    // nextDown(f32::MAX) = 2^128 - 2 * 2^104
    const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;
    test!(SECOND_LARGEST_F32, f32 -> u128, 0xfffffe00000000000000000000000000);
}

fn nan_casts() {
    let nan1 = f64::from_bits(0x7FF0_0001_0000_0001u64);
    let nan2 = f64::from_bits(0x7FF0_0000_0000_0001u64);

    assert!(nan1.is_nan());
    assert!(nan2.is_nan());

    let nan1_32 = nan1 as f32;
    let nan2_32 = nan2 as f32;

    assert!(nan1_32.is_nan());
    assert!(nan2_32.is_nan());
}

fn rounding() {
    // Test cases taken from the library's tests for this feature
    // f16
    assert_eq(2.5f16.round_ties_even(), 2.0f16);
    assert_eq(1.0f16.round_ties_even(), 1.0f16);
    assert_eq(1.3f16.round_ties_even(), 1.0f16);
    assert_eq(1.5f16.round_ties_even(), 2.0f16);
    assert_eq(1.7f16.round_ties_even(), 2.0f16);
    assert_eq(0.0f16.round_ties_even(), 0.0f16);
    assert_eq((-0.0f16).round_ties_even(), -0.0f16);
    assert_eq((-1.0f16).round_ties_even(), -1.0f16);
    assert_eq((-1.3f16).round_ties_even(), -1.0f16);
    assert_eq((-1.5f16).round_ties_even(), -2.0f16);
    assert_eq((-1.7f16).round_ties_even(), -2.0f16);
    // f32
    assert_eq(2.5f32.round_ties_even(), 2.0f32);
    assert_eq(1.0f32.round_ties_even(), 1.0f32);
    assert_eq(1.3f32.round_ties_even(), 1.0f32);
    assert_eq(1.5f32.round_ties_even(), 2.0f32);
    assert_eq(1.7f32.round_ties_even(), 2.0f32);
    assert_eq(0.0f32.round_ties_even(), 0.0f32);
    assert_eq((-0.0f32).round_ties_even(), -0.0f32);
    assert_eq((-1.0f32).round_ties_even(), -1.0f32);
    assert_eq((-1.3f32).round_ties_even(), -1.0f32);
    assert_eq((-1.5f32).round_ties_even(), -2.0f32);
    assert_eq((-1.7f32).round_ties_even(), -2.0f32);
    // f64
    assert_eq(2.5f64.round_ties_even(), 2.0f64);
    assert_eq(1.0f64.round_ties_even(), 1.0f64);
    assert_eq(1.3f64.round_ties_even(), 1.0f64);
    assert_eq(1.5f64.round_ties_even(), 2.0f64);
    assert_eq(1.7f64.round_ties_even(), 2.0f64);
    assert_eq(0.0f64.round_ties_even(), 0.0f64);
    assert_eq((-0.0f64).round_ties_even(), -0.0f64);
    assert_eq((-1.0f64).round_ties_even(), -1.0f64);
    assert_eq((-1.3f64).round_ties_even(), -1.0f64);
    assert_eq((-1.5f64).round_ties_even(), -2.0f64);
    assert_eq((-1.7f64).round_ties_even(), -2.0f64);
    // f128
    assert_eq(2.5f128.round_ties_even(), 2.0f128);
    assert_eq(1.0f128.round_ties_even(), 1.0f128);
    assert_eq(1.3f128.round_ties_even(), 1.0f128);
    assert_eq(1.5f128.round_ties_even(), 2.0f128);
    assert_eq(1.7f128.round_ties_even(), 2.0f128);
    assert_eq(0.0f128.round_ties_even(), 0.0f128);
    assert_eq((-0.0f128).round_ties_even(), -0.0f128);
    assert_eq((-1.0f128).round_ties_even(), -1.0f128);
    assert_eq((-1.3f128).round_ties_even(), -1.0f128);
    assert_eq((-1.5f128).round_ties_even(), -2.0f128);
    assert_eq((-1.7f128).round_ties_even(), -2.0f128);

    assert_eq!(3.8f16.floor(), 3.0f16);
    assert_eq!((-1.1f16).floor(), -2.0f16);
    assert_eq!(3.8f32.floor(), 3.0f32);
    assert_eq!((-1.1f32).floor(), -2.0f32);
    assert_eq!(3.8f64.floor(), 3.0f64);
    assert_eq!((-1.1f64).floor(), -2.0f64);
    assert_eq!(3.8f128.floor(), 3.0f128);
    assert_eq!((-1.1f128).floor(), -2.0f128);

    assert_eq!(3.8f16.ceil(), 4.0f16);
    assert_eq!((-2.3f16).ceil(), -2.0f16);
    assert_eq!(3.8f32.ceil(), 4.0f32);
    assert_eq!((-2.3f32).ceil(), -2.0f32);
    assert_eq!(3.8f64.ceil(), 4.0f64);
    assert_eq!((-2.3f64).ceil(), -2.0f64);
    assert_eq!(3.8f128.ceil(), 4.0f128);
    assert_eq!((-2.3f128).ceil(), -2.0f128);

    assert_eq!(0.1f16.trunc(), 0.0f16);
    assert_eq!((-0.1f16).trunc(), 0.0f16);
    assert_eq!(0.1f32.trunc(), 0.0f32);
    assert_eq!((-0.1f32).trunc(), 0.0f32);
    assert_eq!(0.1f64.trunc(), 0.0f64);
    assert_eq!((-0.1f64).trunc(), 0.0f64);
    assert_eq!(0.1f128.trunc(), 0.0f128);
    assert_eq!((-0.1f128).trunc(), 0.0f128);

    assert_eq!(3.3_f16.round(), 3.0);
    assert_eq!(2.5_f16.round(), 3.0);
    assert_eq!(3.3_f32.round(), 3.0);
    assert_eq!(2.5_f32.round(), 3.0);
    assert_eq!(3.9_f64.round(), 4.0);
    assert_eq!(2.5_f64.round(), 3.0);
    assert_eq!(3.9_f128.round(), 4.0);
    assert_eq!(2.5_f128.round(), 3.0);
}

fn mul_add() {
    // FIXME(f16_f128): add when supported

    assert_eq!(3.0f32.mul_add(2.0f32, 5.0f32), 11.0);
    assert_eq!(0.0f32.mul_add(-2.0, f32::consts::E), f32::consts::E);
    assert_eq!(3.0f64.mul_add(2.0, 5.0), 11.0);
    assert_eq!(0.0f64.mul_add(-2.0f64, f64::consts::E), f64::consts::E);
    assert_eq!((-3.2f32).mul_add(2.4, f32::NEG_INFINITY), f32::NEG_INFINITY);
    assert_eq!((-3.2f64).mul_add(2.4, f64::NEG_INFINITY), f64::NEG_INFINITY);

    let f = f32::mul_add(
        -0.000000000000000000000000000000000000014728589,
        0.0000037105144,
        0.000000000000000000000000000000000000000000055,
    );
    assert_eq!(f.to_bits(), f32::to_bits(-0.0));
}

pub fn libm() {
    fn ldexp(a: f64, b: i32) -> f64 {
        extern "C" {
            fn ldexp(x: f64, n: i32) -> f64;
        }
        unsafe { ldexp(a, b) }
    }

    assert_eq!(64_f32.sqrt(), 8_f32);
    assert_eq!(64_f64.sqrt(), 8_f64);
    assert_eq!(f32::INFINITY.sqrt(), f32::INFINITY);
    assert_eq!(f64::INFINITY.sqrt(), f64::INFINITY);
    assert_eq!(0.0_f32.sqrt().total_cmp(&0.0), std::cmp::Ordering::Equal);
    assert_eq!(0.0_f64.sqrt().total_cmp(&0.0), std::cmp::Ordering::Equal);
    assert_eq!((-0.0_f32).sqrt().total_cmp(&-0.0), std::cmp::Ordering::Equal);
    assert_eq!((-0.0_f64).sqrt().total_cmp(&-0.0), std::cmp::Ordering::Equal);
    assert!((-5.0_f32).sqrt().is_nan());
    assert!((-5.0_f64).sqrt().is_nan());
    assert!(f32::NEG_INFINITY.sqrt().is_nan());
    assert!(f64::NEG_INFINITY.sqrt().is_nan());
    assert!(f32::NAN.sqrt().is_nan());
    assert!(f64::NAN.sqrt().is_nan());

    assert_approx_eq!(25f32.powi(-2), 0.0016f32);
    assert_approx_eq!(23.2f64.powi(2), 538.24f64);

    assert_approx_eq!(25f32.powf(-2f32), 0.0016f32);
    assert_approx_eq!(400f64.powf(0.5f64), 20f64);

    // Some inputs to powf and powi result in fixed outputs
    // and thus must be exactly equal to that value.
    // C standard says:
    // 1^y = 1 for any y, even a NaN.
    assert_eq!(1f32.powf(10.0), 1.0);
    assert_eq!(1f64.powf(100.0), 1.0);
    assert_eq!(1f32.powf(f32::INFINITY), 1.0);
    assert_eq!(1f64.powf(f64::INFINITY), 1.0);
    assert_eq!(1f32.powf(f32::NAN), 1.0);
    assert_eq!(1f64.powf(f64::NAN), 1.0);

    // f*::NAN is a quiet NAN and should return 1 as well.
    assert_eq!(f32::NAN.powf(0.0), 1.0);
    assert_eq!(f64::NAN.powf(0.0), 1.0);

    assert_eq!(42f32.powf(0.0), 1.0);
    assert_eq!(42f64.powf(0.0), 1.0);
    assert_eq!(f32::INFINITY.powf(0.0), 1.0);
    assert_eq!(f64::INFINITY.powf(0.0), 1.0);

    // f*::NAN is a quiet NAN and should return 1 as well.
    assert_eq!(f32::NAN.powi(0), 1.0);
    assert_eq!(f64::NAN.powi(0), 1.0);

    assert_eq!(10.0f32.powi(0), 1.0);
    assert_eq!(10.0f64.powi(0), 1.0);
    assert_eq!(f32::INFINITY.powi(0), 1.0);
    assert_eq!(f64::INFINITY.powi(0), 1.0);

    assert_eq!((-1f32).powf(f32::INFINITY), 1.0);
    assert_eq!((-1f64).powf(f64::INFINITY), 1.0);
    assert_eq!((-1f32).powf(f32::NEG_INFINITY), 1.0);
    assert_eq!((-1f64).powf(f64::NEG_INFINITY), 1.0);

    // For pow (powf in rust) the C standard says:
    // x^0 = 1 for all x even a sNaN
    // FIXME(#4286): this does not match the behavior of all implementations.
    assert_eq!(SNAN_F32.powf(0.0), 1.0);
    assert_eq!(SNAN_F64.powf(0.0), 1.0);

    // For pown (powi in rust) the C standard says:
    // x^0 = 1 for all x even a sNaN
    // FIXME(#4286): this does not match the behavior of all implementations.
    assert_eq!(SNAN_F32.powi(0), 1.0);
    assert_eq!(SNAN_F64.powi(0), 1.0);

    assert_eq!(0f32.powi(10), 0.0);
    assert_eq!(0f64.powi(100), 0.0);
    assert_eq!(0f32.powi(9), 0.0);
    assert_eq!(0f64.powi(99), 0.0);

    assert_biteq((-0f32).powf(10.0), 0.0, "-0^x = +0 where x is positive");
    assert_biteq((-0f64).powf(100.0), 0.0, "-0^x = +0 where x is positive");
    assert_biteq((-0f32).powf(9.0), -0.0, "-0^x = -0 where x is negative");
    assert_biteq((-0f64).powf(99.0), -0.0, "-0^x = -0 where x is negative");

    assert_biteq((-0f32).powi(10), 0.0, "-0^x = +0 where x is positive");
    assert_biteq((-0f64).powi(100), 0.0, "-0^x = +0 where x is positive");
    assert_biteq((-0f32).powi(9), -0.0, "-0^x = -0 where x is negative");
    assert_biteq((-0f64).powi(99), -0.0, "-0^x = -0 where x is negative");

    assert_approx_eq!(1f32.exp(), f32::consts::E);
    assert_approx_eq!(1f64.exp(), f64::consts::E);
    assert_eq!(0f32.exp(), 1.0);
    assert_eq!(0f64.exp(), 1.0);

    assert_approx_eq!(1f32.exp_m1(), f32::consts::E - 1.0);
    assert_approx_eq!(1f64.exp_m1(), f64::consts::E - 1.0);

    assert_approx_eq!(10f32.exp2(), 1024f32);
    assert_approx_eq!(50f64.exp2(), 1125899906842624f64);
    assert_eq!(0f32.exp2(), 1.0);
    assert_eq!(0f64.exp2(), 1.0);

    assert_approx_eq!(f32::consts::E.ln(), 1f32);
    assert_approx_eq!(f64::consts::E.ln(), 1f64);
    assert_eq!(1f32.ln(), 0.0);
    assert_eq!(1f64.ln(), 0.0);

    assert_approx_eq!(0f32.ln_1p(), 0f32);
    assert_approx_eq!(0f64.ln_1p(), 0f64);

    assert_approx_eq!(10f32.log10(), 1f32);
    assert_approx_eq!(f64::consts::E.log10(), f64::consts::LOG10_E);

    assert_approx_eq!(8f32.log2(), 3f32);
    assert_approx_eq!(f64::consts::E.log2(), f64::consts::LOG2_E);

    #[allow(deprecated)]
    {
        assert_approx_eq!(5.0f32.abs_sub(3.0), 2.0);
        assert_approx_eq!(3.0f64.abs_sub(5.0), 0.0);
    }

    assert_approx_eq!(27.0f32.cbrt(), 3.0f32);
    assert_approx_eq!(27.0f64.cbrt(), 3.0f64);

    assert_approx_eq!(3.0f32.hypot(4.0f32), 5.0f32);
    assert_approx_eq!(3.0f64.hypot(4.0f64), 5.0f64);

    assert_eq!(ldexp(0.65f64, 3i32), 5.2f64);
    assert_eq!(ldexp(1.42, 0xFFFF), f64::INFINITY);
    assert_eq!(ldexp(1.42, -0xFFFF), 0f64);

    // Trigonometric functions.

    assert_eq!(0f32.sin(), 0f32);
    assert_eq!(0f64.sin(), 0f64);
    assert_approx_eq!((f64::consts::PI / 2f64).sin(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.sin().asin(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.sin().asin(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.sinh(), 1.1752012f32);
    assert_approx_eq!(1.0f64.sinh(), 1.1752011936438014f64);
    assert_approx_eq!(2.0f32.asinh(), 1.443635475178810342493276740273105f32);
    assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);

    // Ensure `sin` always returns something that is a valid input for `asin`, and same for
    // `cos` and `acos`.
    let halve_pi_f32 = std::f32::consts::FRAC_PI_2;
    let halve_pi_f64 = std::f64::consts::FRAC_PI_2;
    let pi_f32 = std::f32::consts::PI;
    let pi_f64 = std::f64::consts::PI;
    for _ in 0..64 {
        // sin() should be clamped to [-1, 1] so asin() can never return NaN
        assert!(!halve_pi_f32.sin().asin().is_nan());
        assert!(!halve_pi_f64.sin().asin().is_nan());
        // cos() should be clamped to [-1, 1] so acos() can never return NaN
        assert!(!pi_f32.cos().acos().is_nan());
        assert!(!pi_f64.cos().acos().is_nan());
    }

    assert_eq!(0f32.cos(), 1f32);
    assert_eq!(0f64.cos(), 1f64);
    assert_approx_eq!((f64::consts::PI * 2f64).cos(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.cos().acos(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.cos().acos(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.cosh(), 1.54308f32);
    assert_approx_eq!(1.0f64.cosh(), 1.5430806348152437f64);
    assert_approx_eq!(2.0f32.acosh(), 1.31695789692481670862504634730796844f32);
    assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);

    assert_approx_eq!(1.0f32.tan(), 1.557408f32);
    assert_approx_eq!(1.0f64.tan(), 1.5574077246549023f64);
    assert_approx_eq!(1.0_f32, 1.0_f32.tan().atan());
    assert_approx_eq!(1.0_f64, 1.0_f64.tan().atan());
    assert_approx_eq!(1.0f32.atan2(2.0f32), 0.46364761f32);
    assert_approx_eq!(1.0f32.atan2(2.0f32), 0.46364761f32);

    assert_approx_eq!(
        1.0f32.tanh(),
        (1.0 - f32::consts::E.powi(-2)) / (1.0 + f32::consts::E.powi(-2))
    );
    assert_approx_eq!(
        1.0f64.tanh(),
        (1.0 - f64::consts::E.powi(-2)) / (1.0 + f64::consts::E.powi(-2))
    );
    assert_approx_eq!(0.5f32.atanh(), 0.54930614433405484569762261846126285f32);
    assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);

    assert_approx_eq!(5.0f32.gamma(), 24.0);
    assert_approx_eq!(5.0f64.gamma(), 24.0);
    assert_approx_eq!((-0.5f32).gamma(), (-2.0) * f32::consts::PI.sqrt());
    assert_approx_eq!((-0.5f64).gamma(), (-2.0) * f64::consts::PI.sqrt());

    assert_eq!(2.0f32.ln_gamma(), (0.0, 1));
    assert_eq!(2.0f64.ln_gamma(), (0.0, 1));
    // Gamma(-0.5) = -2*sqrt(π)
    let (val, sign) = (-0.5f32).ln_gamma();
    assert_approx_eq!(val, (2.0 * f32::consts::PI.sqrt()).ln());
    assert_eq!(sign, -1);
    let (val, sign) = (-0.5f64).ln_gamma();
    assert_approx_eq!(val, (2.0 * f64::consts::PI.sqrt()).ln());
    assert_eq!(sign, -1);

    assert_approx_eq!(1.0f32.erf(), 0.84270079294971486934122063508260926f32);
    assert_approx_eq!(1.0f64.erf(), 0.84270079294971486934122063508260926f64);
    assert_approx_eq!(1.0f32.erfc(), 0.15729920705028513065877936491739074f32);
    assert_approx_eq!(1.0f64.erfc(), 0.15729920705028513065877936491739074f64);
}

fn test_fast() {
    use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, frem_fast, fsub_fast};

    #[inline(never)]
    pub fn test_operations_f16(a: f16, b: f16) {
        // make sure they all map to the correct operation
        unsafe {
            assert_approx_eq!(fadd_fast(a, b), a + b);
            assert_approx_eq!(fsub_fast(a, b), a - b);
            assert_approx_eq!(fmul_fast(a, b), a * b);
            assert_approx_eq!(fdiv_fast(a, b), a / b);
            assert_approx_eq!(frem_fast(a, b), a % b);
        }
    }

    #[inline(never)]
    pub fn test_operations_f32(a: f32, b: f32) {
        // make sure they all map to the correct operation
        unsafe {
            assert_approx_eq!(fadd_fast(a, b), a + b);
            assert_approx_eq!(fsub_fast(a, b), a - b);
            assert_approx_eq!(fmul_fast(a, b), a * b);
            assert_approx_eq!(fdiv_fast(a, b), a / b);
            assert_approx_eq!(frem_fast(a, b), a % b);
        }
    }

    #[inline(never)]
    pub fn test_operations_f64(a: f64, b: f64) {
        // make sure they all map to the correct operation
        unsafe {
            assert_approx_eq!(fadd_fast(a, b), a + b);
            assert_approx_eq!(fsub_fast(a, b), a - b);
            assert_approx_eq!(fmul_fast(a, b), a * b);
            assert_approx_eq!(fdiv_fast(a, b), a / b);
            assert_approx_eq!(frem_fast(a, b), a % b);
        }
    }

    #[inline(never)]
    pub fn test_operations_f128(a: f128, b: f128) {
        // make sure they all map to the correct operation
        unsafe {
            assert_approx_eq!(fadd_fast(a, b), a + b);
            assert_approx_eq!(fsub_fast(a, b), a - b);
            assert_approx_eq!(fmul_fast(a, b), a * b);
            assert_approx_eq!(fdiv_fast(a, b), a / b);
            assert_approx_eq!(frem_fast(a, b), a % b);
        }
    }

    test_operations_f16(11., 2.);
    test_operations_f16(10., 15.);
    test_operations_f32(11., 2.);
    test_operations_f32(10., 15.);
    test_operations_f64(1., 2.);
    test_operations_f64(10., 5.);
    test_operations_f128(1., 2.);
    test_operations_f128(10., 5.);
}

fn test_algebraic() {
    use std::intrinsics::{
        fadd_algebraic, fdiv_algebraic, fmul_algebraic, frem_algebraic, fsub_algebraic,
    };

    #[inline(never)]
    pub fn test_operations_f16(a: f16, b: f16) {
        // make sure they all map to the correct operation
        assert_approx_eq!(fadd_algebraic(a, b), a + b);
        assert_approx_eq!(fsub_algebraic(a, b), a - b);
        assert_approx_eq!(fmul_algebraic(a, b), a * b);
        assert_approx_eq!(fdiv_algebraic(a, b), a / b);
        assert_approx_eq!(frem_algebraic(a, b), a % b);
    }

    #[inline(never)]
    pub fn test_operations_f32(a: f32, b: f32) {
        // make sure they all map to the correct operation
        assert_approx_eq!(fadd_algebraic(a, b), a + b);
        assert_approx_eq!(fsub_algebraic(a, b), a - b);
        assert_approx_eq!(fmul_algebraic(a, b), a * b);
        assert_approx_eq!(fdiv_algebraic(a, b), a / b);
        assert_approx_eq!(frem_algebraic(a, b), a % b);
    }

    #[inline(never)]
    pub fn test_operations_f64(a: f64, b: f64) {
        // make sure they all map to the correct operation
        assert_approx_eq!(fadd_algebraic(a, b), a + b);
        assert_approx_eq!(fsub_algebraic(a, b), a - b);
        assert_approx_eq!(fmul_algebraic(a, b), a * b);
        assert_approx_eq!(fdiv_algebraic(a, b), a / b);
        assert_approx_eq!(frem_algebraic(a, b), a % b);
    }

    #[inline(never)]
    pub fn test_operations_f128(a: f128, b: f128) {
        // make sure they all map to the correct operation
        assert_approx_eq!(fadd_algebraic(a, b), a + b);
        assert_approx_eq!(fsub_algebraic(a, b), a - b);
        assert_approx_eq!(fmul_algebraic(a, b), a * b);
        assert_approx_eq!(fdiv_algebraic(a, b), a / b);
        assert_approx_eq!(frem_algebraic(a, b), a % b);
    }

    test_operations_f16(11., 2.);
    test_operations_f16(10., 15.);
    test_operations_f32(11., 2.);
    test_operations_f32(10., 15.);
    test_operations_f64(1., 2.);
    test_operations_f64(10., 5.);
    test_operations_f128(1., 2.);
    test_operations_f128(10., 5.);
}

fn test_fmuladd() {
    use std::intrinsics::{fmuladdf32, fmuladdf64};

    // FIXME(f16_f128): add when supported

    #[inline(never)]
    pub fn test_operations_f32(a: f32, b: f32, c: f32) {
        assert_approx_eq!(unsafe { fmuladdf32(a, b, c) }, a * b + c);
    }

    #[inline(never)]
    pub fn test_operations_f64(a: f64, b: f64, c: f64) {
        assert_approx_eq!(unsafe { fmuladdf64(a, b, c) }, a * b + c);
    }

    test_operations_f32(0.1, 0.2, 0.3);
    test_operations_f64(1.1, 1.2, 1.3);
}

/// `min` and `max` on equal arguments are non-deterministic.
fn test_min_max_nondet() {
    /// Ensure that if we call the closure often enough, we see both `true` and `false.`
    #[track_caller]
    fn ensure_both(f: impl Fn() -> bool) {
        let rounds = 16;
        let first = f();
        for _ in 1..rounds {
            if f() != first {
                // We saw two different values!
                return;
            }
        }
        // We saw the same thing N times.
        panic!("expected non-determinism, got {rounds} times the same result: {first:?}");
    }

    ensure_both(|| f16::min(0.0, -0.0).is_sign_positive());
    ensure_both(|| f16::max(0.0, -0.0).is_sign_positive());
    ensure_both(|| f32::min(0.0, -0.0).is_sign_positive());
    ensure_both(|| f32::max(0.0, -0.0).is_sign_positive());
    ensure_both(|| f64::min(0.0, -0.0).is_sign_positive());
    ensure_both(|| f64::max(0.0, -0.0).is_sign_positive());
    ensure_both(|| f128::min(0.0, -0.0).is_sign_positive());
    ensure_both(|| f128::max(0.0, -0.0).is_sign_positive());
}

fn test_non_determinism() {
    use std::intrinsics::{
        fadd_algebraic, fadd_fast, fdiv_algebraic, fdiv_fast, fmul_algebraic, fmul_fast,
        frem_algebraic, frem_fast, fsub_algebraic, fsub_fast,
    };
    use std::{f32, f64};
    // TODO: Also test powi and powf when the non-determinism is implemented for them

    /// Ensure that the operation is non-deterministic
    #[track_caller]
    fn ensure_nondet<T: PartialEq + std::fmt::Debug>(f: impl Fn() -> T) {
        let rounds = 16;
        let first = f();
        for _ in 1..rounds {
            if f() != first {
                // We saw two different values!
                return;
            }
        }
        // We saw the same thing N times.
        panic!("expected non-determinism, got {rounds} times the same result: {first:?}");
    }

    macro_rules! test_operations_f {
        ($a:expr, $b:expr) => {
            ensure_nondet(|| fadd_algebraic($a, $b));
            ensure_nondet(|| fsub_algebraic($a, $b));
            ensure_nondet(|| fmul_algebraic($a, $b));
            ensure_nondet(|| fdiv_algebraic($a, $b));
            ensure_nondet(|| frem_algebraic($a, $b));

            unsafe {
                ensure_nondet(|| fadd_fast($a, $b));
                ensure_nondet(|| fsub_fast($a, $b));
                ensure_nondet(|| fmul_fast($a, $b));
                ensure_nondet(|| fdiv_fast($a, $b));
                ensure_nondet(|| frem_fast($a, $b));
            }
        };
    }

    pub fn test_operations_f16(a: f16, b: f16) {
        test_operations_f!(a, b);
    }
    pub fn test_operations_f32(a: f32, b: f32) {
        test_operations_f!(a, b);
        // FIXME: temporarily disabled as it breaks std tests.
        // ensure_nondet(|| a.log(b));
        // ensure_nondet(|| a.exp());
        // ensure_nondet(|| 10f32.exp2());
        // ensure_nondet(|| f32::consts::E.ln());
        // ensure_nondet(|| 1f32.ln_1p());
        // ensure_nondet(|| 10f32.log10());
        // ensure_nondet(|| 8f32.log2());
        // ensure_nondet(|| 27.0f32.cbrt());
        // ensure_nondet(|| 3.0f32.hypot(4.0f32));
        // ensure_nondet(|| 1f32.sin());
        // ensure_nondet(|| 0f32.cos());
        // // On i686-pc-windows-msvc , these functions are implemented by calling the `f64` version,
        // // which means the little rounding errors Miri introduces are discard by the cast down to `f32`.
        // // Just skip the test for them.
        // if !cfg!(all(target_os = "windows", target_env = "msvc", target_arch = "x86")) {
        //     ensure_nondet(|| 1.0f32.tan());
        //     ensure_nondet(|| 1.0f32.asin());
        //     ensure_nondet(|| 5.0f32.acos());
        //     ensure_nondet(|| 1.0f32.atan());
        //     ensure_nondet(|| 1.0f32.atan2(2.0f32));
        //     ensure_nondet(|| 1.0f32.sinh());
        //     ensure_nondet(|| 1.0f32.cosh());
        //     ensure_nondet(|| 1.0f32.tanh());
        // }
        // ensure_nondet(|| 1.0f32.asinh());
        // ensure_nondet(|| 2.0f32.acosh());
        // ensure_nondet(|| 0.5f32.atanh());
        // ensure_nondet(|| 5.0f32.gamma());
        // ensure_nondet(|| 5.0f32.ln_gamma());
        // ensure_nondet(|| 5.0f32.erf());
        // ensure_nondet(|| 5.0f32.erfc());
    }
    pub fn test_operations_f64(a: f64, b: f64) {
        test_operations_f!(a, b);
        // FIXME: temporarily disabled as it breaks std tests.
        // ensure_nondet(|| a.log(b));
        // ensure_nondet(|| a.exp());
        // ensure_nondet(|| 50f64.exp2());
        // ensure_nondet(|| 3f64.ln());
        // ensure_nondet(|| 1f64.ln_1p());
        // ensure_nondet(|| f64::consts::E.log10());
        // ensure_nondet(|| f64::consts::E.log2());
        // ensure_nondet(|| 27.0f64.cbrt());
        // ensure_nondet(|| 3.0f64.hypot(4.0f64));
        // ensure_nondet(|| 1f64.sin());
        // ensure_nondet(|| 0f64.cos());
        // ensure_nondet(|| 1.0f64.tan());
        // ensure_nondet(|| 1.0f64.asin());
        // ensure_nondet(|| 5.0f64.acos());
        // ensure_nondet(|| 1.0f64.atan());
        // ensure_nondet(|| 1.0f64.atan2(2.0f64));
        // ensure_nondet(|| 1.0f64.sinh());
        // ensure_nondet(|| 1.0f64.cosh());
        // ensure_nondet(|| 1.0f64.tanh());
        // ensure_nondet(|| 1.0f64.asinh());
        // ensure_nondet(|| 3.0f64.acosh());
        // ensure_nondet(|| 0.5f64.atanh());
        // ensure_nondet(|| 5.0f64.gamma());
        // ensure_nondet(|| 5.0f64.ln_gamma());
        // ensure_nondet(|| 5.0f64.erf());
        // ensure_nondet(|| 5.0f64.erfc());
    }
    pub fn test_operations_f128(a: f128, b: f128) {
        test_operations_f!(a, b);
    }

    test_operations_f16(5., 7.);
    test_operations_f32(12., 5.);
    test_operations_f64(19., 11.);
    test_operations_f128(25., 18.);
}
