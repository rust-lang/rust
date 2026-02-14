//! This module is a port of the casting tests in the miri test suite at
//! https://github.com/rust-lang/rust/blob/fef627b1ebdc7369ddf8a4031a5d25733ac1fb34/src/tools/miri/tests/pass/float.rs

use std::any::type_name;
use std::cmp::min;
use std::fmt::{Debug, Display};
use std::hint::black_box;
use std::{f32, f64};

use super::TestableFloat;

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
                    unsafe { self.to_int_unchecked() }
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
#[track_caller]
fn assert_biteq<F: TestableFloat>(a: F, b: F, msg: impl Display) {
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
#[track_caller]
fn assert_feq<F: TestableFloat>(a: F, b: F, msg: impl Display) {
    let ab = a.to_bits();
    let bb = b.to_bits();
    let tname = type_name::<F>();
    let width = (2 + F::BITS / 4) as usize;
    assert_eq_msg::<F>(a, b, format_args!("({ab:#0width$x} != {bb:#0width$x}) {tname}: {msg}"));
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

        let fbits = <$fty as TestableFloat>::BITS;
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
        type F2Int = <$f2 as TestableFloat>::Int;

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
        let min_neg_sub_casted =
            <$f1>::from_bits(0x1 | 1 << (<$f1 as TestableFloat>::BITS - 1)) as $f2;

        if <$f1 as TestableFloat>::BITS > <$f2 as TestableFloat>::BITS {
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

macro_rules! int_float_generic_casts {
    ($f:ty) => {
        test_ftoi_itof! { f: $f, i: i8, imin_f: -128.0, imax_f: 127.0 };
        test_ftoi_itof! { f: $f, i: u8, imin_f: 0.0, imax_f: 255.0 };
        test_ftoi_itof! { f: $f, i: i16, imin_f: -32_768.0, imax_f: 32_767.0 };
        test_ftoi_itof! { f: $f, i: u16, imin_f: 0.0, imax_f: 65_535.0 };
        test_ftoi_itof! { f: $f, i: i32, imin_f: -2_147_483_648.0, imax_f: 2_147_483_647.0 };
        test_ftoi_itof! { f: $f, i: u32, imin_f: 0.0, imax_f: 4_294_967_295.0 };
        test_ftoi_itof! {
            f: $f,
            i: i64,
            imin_f: -9_223_372_036_854_775_808.0,
            imax_f: 9_223_372_036_854_775_807.0
        };
        test_ftoi_itof! { f: $f, i: u64, imin_f: 0.0, imax_f: 18_446_744_073_709_551_615.0 };
        test_ftoi_itof! {
            f: $f,
            i: i128,
            imin_f: -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
            imax_f: 170_141_183_460_469_231_731_687_303_715_884_105_727.0,
        };
        test_ftoi_itof! {
            f: $f,
            i: u128,
            imin_f: 0.0,
            imax_f: 340_282_366_920_938_463_463_374_607_431_768_211_455.0
        };
    };
}

#[test]
fn int_f16_generic_casts() {
    int_float_generic_casts!(f16);
}

#[test]
fn int_f32_generic_casts() {
    int_float_generic_casts!(f32);
}

#[test]
fn int_f64_generic_casts() {
    int_float_generic_casts!(f64);
}

#[test]
fn int_f128_generic_casts() {
    int_float_generic_casts!(f128);
}

#[test]
fn int_float_spot_checks() {
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
}

#[test]
fn float_float_generic_tests() {
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
}

#[test]
fn float_float_spot_checks() {
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

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => (
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run-time {} -> {}", stringify!($src_ty), stringify!($dest_ty));

        {
            const VAL: $src_ty = $val;
            assert_eq!(const { VAL as $dest_ty }, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
    );

    ($fval:expr, f>16 -> $ity:ident, $ival:expr) => (
        test!($fval, f32 -> $ity, $ival);
        test!($fval, f64 -> $ity, $ival);
        test!($fval, f128 -> $ity, $ival);
    );
    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test!($fval, f16 -> $ity, $ival);
        test!($fval, f>16 -> $ity, $ival);
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
        common_fptoi_tests!(f16 -> $($ity)+);
        common_fptoi_tests!(f32 -> $($ity)+);
        common_fptoi_tests!(f64 -> $($ity)+);
        common_fptoi_tests!(f128 -> $($ity)+);
    })
}

macro_rules! fptoui_tests {
    ($fty: ident -> $($ity: ident)+) => (
    #[allow(overflowing_literals)]
    { $(
        test!(-0., $fty -> $ity, 0);
        test!(-$fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.99999994, $fty -> $ity, 0);
        test!(-1., $fty -> $ity, 0);
        test!(-100., $fty -> $ity, 0);
        test!(-1e50, $fty -> $ity, 0);
        test!(-1e130, $fty -> $ity, 0);
    )+ });

    (f* -> $($ity:ident)+) => ({
        fptoui_tests!(f16 -> $($ity)+);
        fptoui_tests!(f32 -> $($ity)+);
        fptoui_tests!(f64 -> $($ity)+);
        fptoui_tests!(f128 -> $($ity)+);
    })
}

#[test]
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
    test!(-2147483648., f>16 -> i32, i32::MIN);
    // 2147483648. is i32::MAX rounded up
    test!(2147483648., f32 -> i32, 2147483647);
    // With 24 significand bits, floats with magnitude in [2^30 + 1, 2^31] are rounded to
    // multiples of 2^7. Therefore, nextDown(round(i32::MAX)) is 2^31 - 128:
    test!(2147483520., f32 -> i32, 2147483520);
    // Similarly, nextUp(i32::MIN) is i32::MIN + 2^8 and nextDown(i32::MIN) is i32::MIN - 2^7
    test!(-2147483904., f>16 -> i32, i32::MIN);
    test!(-2147483520., f>16 -> i32, -2147483520);

    // # u32
    // round(MAX) and nextUp(round(MAX))
    test!(4294967040., f>16 -> u32, 4294967040);
    test!(4294967296., f>16 -> u32, 4294967295);

    // # u128
    // float->int:
    test!(f32::MAX, f32 -> u128, 0xffffff00000000000000000000000000);
    // nextDown(f32::MAX) = 2^128 - 2 * 2^104
    const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;
    test!(SECOND_LARGEST_F32, f32 -> u128, 0xfffffe00000000000000000000000000);
}

#[test]
fn nan_casts() {
    let nan1 = f64::from_bits(0x7FF0_0001_0000_0001u64);
    let nan2 = f64::from_bits(0x7FF0_0000_0000_0001u64);

    assert!(nan1.is_nan());
    assert!(nan2.is_nan());

    let nan1_128 = nan1 as f128;
    let nan2_128 = nan2 as f128;

    assert!(nan1_128.is_nan());
    assert!(nan2_128.is_nan());

    let nan1_32 = nan1 as f32;
    let nan2_32 = nan2 as f32;

    assert!(nan1_32.is_nan());
    assert!(nan2_32.is_nan());

    let nan1_16 = nan1 as f16;
    let nan2_16 = nan2 as f16;

    assert!(nan1_16.is_nan());
    assert!(nan2_16.is_nan());
}
