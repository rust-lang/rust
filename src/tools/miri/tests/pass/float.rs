#![feature(stmt_expr_attributes)]
#![feature(float_gamma)]
#![feature(core_intrinsics)]
#![feature(f128)]
#![feature(f16)]
#![allow(arithmetic_overflow)]
#![allow(internal_features)]

use std::any::type_name;
use std::convert::TryFrom;
use std::fmt::{Debug, LowerHex};
use std::hint::black_box;
use std::{f32, f64};

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6, "{} is not approximately equal to {}", *a, *b);
    }};
}

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
}

trait Float: Copy + PartialEq + Debug {
    type Int: Int;
    const BITS: u32;
    const EXPONENT_BITS: u32;
    const SIGNIFICAND_BITS: u32 = Self::BITS - Self::EXPONENT_BITS - 1;
    /// The maximum value of the exponent
    const EXPONENT_MAX: u32 = (1 << Self::EXPONENT_BITS) - 1;
    /// The exponent bias value
    const EXPONENT_BIAS: u32 = Self::EXPONENT_MAX >> 1;

    const ZERO: Self;
    const FIVE: Self;

    fn to_bits(self) -> Self::Int;
}

macro_rules! impl_float {
    ($ty:ty, $ity:ty, $bits:literal, $sig_bits:literal) => {
        impl Float for $ty {
            type Int = $ity;
            const BITS: u32 = $bits;
            const EXPONENT_BITS: u32 = $sig_bits;
            const ZERO: Self = 0.0;
            const FIVE: Self = 5.0;

            fn to_bits(self) -> Self::Int {
                self.to_bits()
            }
        }
    };
}

trait Int: Copy + PartialEq + Debug + LowerHex {
    /// Unsigned integer of the same size, or self if already unsigned
    type Unsigned;
    const SIGNED: bool;
    const BITS: u32;
    const ZERO: Self;
    /// Negative one if signed, zero otherwise
    const NEG_ONE: Self;
    /// Negative five if signed, zero otherwise
    const NEG_FIVE: Self;
}

macro_rules! impl_int {
    ($($sty:ty, $uty:ty);+) => {
        $(
            impl Int for $sty {
                type Unsigned = $uty;
                const SIGNED: bool = true;
                const BITS: u32 = Self::BITS;
                const ZERO: Self = 0;
                const NEG_ONE: Self = -1;
                const NEG_FIVE: Self = -5;
            }

            impl Int for $uty {
                type Unsigned = Self;
                const SIGNED: bool = false;
                const BITS: u32 = Self::BITS;
                const NEG_ONE: Self = 0;
                const NEG_FIVE: Self = 0;
                const ZERO: Self = 0;
            }
        )+
    };
}

impl_float!(f32, u32, 32, 8);
impl_float!(f64, u64, 64, 11);
impl_int!(i8, u8; i16, u16; i32, u32; i64, u64; i128, u128);

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

float_to_int!(f32 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
float_to_int!(f64 => i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

/// Test this cast both via `as` and via `approx_unchecked` (i.e., it must not saturate).
#[track_caller]
#[inline(never)]
fn test_both_cast<F, I>(x: F, y: I, msg: &str)
where
    F: FloatToInt<I>,
    I: PartialEq + Debug,
{
    assert_eq!(x.cast(), y, "{} -> {}: {msg}", type_name::<F>(), type_name::<I>());
    assert_eq!(
        unsafe { x.cast_unchecked() },
        y,
        "{} -> {}: {msg}",
        stringify!($fty),
        stringify!($ity)
    );
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
fn assert_eq_msg<T: PartialEq + Debug>(x: T, y: T, msg: &str) {
    assert_eq!(x, y, "{msg}");
}

/// Check that two floats have equality
fn assert_biteq<F: Float>(a: F, b: F, msg: &str) {
    let ab = a.to_bits();
    let bb = b.to_bits();
    let msg = format!(
        "({ab:#0width$x} != {bb:#0width$x}) {}: {msg}",
        type_name::<F>(),
        width = (2 + F::Int::BITS / 4) as usize
    );
    assert_eq_msg::<F::Int>(ab, bb, &msg);
}

/// Check that floats have bitwise equality
fn assert_feq<F: Float>(a: F, b: F, msg: &str) {
    let ab = a.to_bits();
    let bb = b.to_bits();
    let msg = format!(
        "({ab:#0width$x} != {bb:#0width$x}) {}: {msg}",
        type_name::<F>(),
        width = (2 + F::Int::BITS / 4) as usize
    );
    assert_eq_msg::<F>(a, b, &msg);
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
    // FIXME(f16_f128): add when constants and `is_infinite` are available
    assert!((5.0_f32 / 0.0).is_infinite());
    assert_ne!({ 5.0_f32 / 0.0 }, { -5.0_f32 / 0.0 });
    assert!((5.0_f64 / 0.0).is_infinite());
    assert_ne!({ 5.0_f64 / 0.0 }, { 5.0_f64 / -0.0 });
    assert_ne!(f32::NAN, f32::NAN);
    assert_ne!(f64::NAN, f64::NAN);

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

    // FIXME(f16_f128): add when `abs` is available
    assert_eq!((-1.0f32).abs(), 1.0f32);
    assert_eq!(34.2f64.abs(), 34.2f64);
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
            test_both_cast::<$fty, $ity>(f, i, msg);
        }

        /// Unrepresentable values only get tested with `as` casting, not unchecked
        fn assert_ftoi_unrep(f: $fty, i: $ity, msg: &str) {
            let msg = format!("{} -> {}: {msg}", stringify!($fty), stringify!($ity));
            assert_eq_msg::<$ity>(f as $ity, i, &msg);
        }

        /// Int to float checks
        fn assert_itof(i: $ity, f: $fty, msg: &str) {
            let itof_msg = format!("{} -> {}: {msg}", stringify!($ity), stringify!($fty));
            assert_eq_msg::<$fty>(i as $fty, f, &itof_msg);
        }

        /// Check both float to int and int to float
        fn assert_bidir(f: $fty, i: $ity, msg: &str) {
            assert_ftoi(f, i, msg);
            assert_itof(i, f, msg);
        }

        /// Check both float to int and int to float
        fn assert_bidir_unrep(f: $fty, i: $ity, msg: &str) {
            assert_ftoi_unrep(f, i, msg);
            assert_itof(i, f, msg);
        }

        let fbits = <$fty>::BITS;
        let ibits = <$ity>::BITS;
        let imax = <$ity>::MAX;
        let imin = <$ity>::MIN;
        let isigned = <$ity>::SIGNED;

        #[allow(overflowing_literals)]
        let imin_f = $imin_f;
        #[allow(overflowing_literals)]
        let imax_f = $imax_f;

        // Skip unchecked cast if negative numbers are unrepresentable
        let assert_ftoi_neg = if isigned { assert_ftoi } else { assert_ftoi_unrep };
        // Skip unchecked cast when int min/max would be unrepresentable
        let assert_ftoi_big = if ibits < fbits { assert_ftoi } else { assert_ftoi_unrep };
        let assert_bidir_big = if ibits < fbits { assert_bidir } else { assert_bidir_unrep };

        // Near zero representations
        assert_bidir(0.0, 0, "zero");
        assert_ftoi(-0.0, 0, "negative zero");
        assert_ftoi_neg(-0.99999999999999999999999999, <$ity>::NEG_ONE, "near -1");
        assert_ftoi(<$fty>::from_bits(0x1), 0, "min subnormal");
        assert_ftoi(<$fty>::from_bits(0x1 | 1 << (fbits - 1)), 0, "min neg subnormal");

        // Spot checks
        assert_ftoi(0.9, 0, "0.9");
        assert_ftoi_neg(-0.9, 0, "-0.9");
        assert_ftoi(1.1, 1, "1.1");
        assert_ftoi_neg(-1.1, <$ity>::NEG_ONE, "-1.1");
        assert_ftoi(1.9, 1, "1.9");
        assert_ftoi_neg(-1.9, <$ity>::NEG_ONE, "-1.9");
        assert_ftoi(5.0, 5, "5.0");
        assert_ftoi_neg(-5.0, <$ity>::NEG_FIVE, "-5.0");
        assert_ftoi(5.9, 5, "5.0");
        assert_ftoi_neg(-5.9, <$ity>::NEG_FIVE, "-5.0");

        // Half representations should always fit perfectly
        let half_i_max =
            <$ity>::try_from((!(0 as <$ity as Int>::Unsigned) >> (ibits / 2)) + 1).unwrap();
        let half_i_min = <$ity>::ZERO.saturating_sub(half_i_max);
        assert_bidir(half_i_max as $fty, half_i_max, "half int max");
        assert_bidir(half_i_min as $fty, half_i_min, "half int min");

        // Integer limits
        assert_bidir_big(imax_f, imax, "i max");
        assert_bidir_big(imin_f, imin, "i min");
        assert_ftoi_big(imax_f + 0.99, <$ity>::MAX, "slightly above i max");
        assert_ftoi_neg(imin_f - 0.99, <$ity>::MIN, "slightly below i min");

        if fbits == ibits {
            // Signed integers have one fewer bit of range
            let adj = if isigned { 1 } else { 0 };

            let exp_overflow = 1 << (<$fty>::EXPONENT_BITS - 1 - adj);
            let exp_max = (1 << <$fty>::EXPONENT_BITS - adj) - 1;

            let max_le_imax = imax - <$ity>::try_from(exp_overflow).unwrap();

            // Max value that can be represented below I::MAX
            assert_bidir(
                max_le_imax as $fty,
                imax - (exp_max as $ity),
                "max representable near int max",
            );

            // The next value up is unrepresentable
            assert_bidir_unrep((max_le_imax + 1) as $fty, imax, "one more than max representable");

            if isigned {
                let min_ge_imin = imin + (exp_overflow as $ity);
                assert_bidir(min_ge_imin as $fty, imin, "min representable near int min");
            }
        }

        if fbits >= ibits {
            // Max representable power of 10
            let pow10_max = (10 as $ity).pow(imax.ilog10());
            assert_bidir(pow10_max as $fty, pow10_max, "pow10 max");

            // Float limits
            assert_ftoi_unrep(<$fty>::MAX, imax, "f max");
            assert_ftoi_unrep(<$fty>::MIN, imin, "f min");
        }

        if ibits > fbits * 2 && !isigned {
            // Unsigned max is enough to saturate only when more than 2x bit width
            assert_itof(imax, <$fty>::INFINITY, "imax, ibits > fbits * 2");
        } else {
            // Signed values and anything smaller still fit
            assert_itof(imax, imax_f, "imax, ibits > fbits * 2");
            assert_itof(imin, imin_f, "imin, ibits > fbits * 2");
        }

        // Float limits
        assert_ftoi_unrep(<$fty>::INFINITY, imax, "f inf");
        assert_ftoi_unrep(<$fty>::NEG_INFINITY, imin, "f neg inf");
        assert_ftoi_unrep(<$fty>::NAN, 0, "f nan");
        assert_ftoi_unrep(-<$fty>::NAN, 0, "f neg nan");
    }};
}

macro_rules! test_f_to_f {
    (
        f1: $f1:ty,
        f2: $f2:ty $(,)?
    ) => {{
        type F2Int = <$f2 as Float>::Int;

        let f1zero = <$f1>::ZERO;
        let f2zero = <$f2>::ZERO;

        assert_biteq((f1zero as $f2), f2zero, "0.0");
        assert_biteq(((-f1zero) as $f2), (-f2zero), "-0.0");
        assert_biteq((<$f1>::FIVE as $f2), <$f2>::FIVE, "5.0");
        assert_biteq(((-<$f1>::FIVE) as $f2), (-<$f2>::FIVE), "-5.0");

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

/// Many of these test values are taken from
/// https://github.com/WebAssembly/testsuite/blob/master/conversions.wast.
fn casts() {
    /* int <-> float generic tests */

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

    test_f_to_f! { f1: f32, f2: f64 };
    test_f_to_f! { f1: f64, f2: f32 };

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

    assert_eq!(3.8f32.floor(), 3.0f32);
    assert_eq!((-1.1f64).floor(), -2.0f64);

    assert_eq!((-2.3f32).ceil(), -2.0f32);
    assert_eq!(3.8f64.ceil(), 4.0f64);

    assert_eq!(0.1f32.trunc(), 0.0f32);
    assert_eq!((-0.1f64).trunc(), 0.0f64);

    assert_eq!(3.3_f32.round(), 3.0);
    assert_eq!(2.5_f32.round(), 3.0);
    assert_eq!(3.9_f64.round(), 4.0);
    assert_eq!(2.5_f64.round(), 3.0);
}

fn mul_add() {
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

    assert_approx_eq!(64f32.sqrt(), 8f32);
    assert_approx_eq!(64f64.sqrt(), 8f64);
    assert!((-5.0_f32).sqrt().is_nan());
    assert!((-5.0_f64).sqrt().is_nan());

    assert_approx_eq!(25f32.powi(-2), 0.0016f32);
    assert_approx_eq!(23.2f64.powi(2), 538.24f64);

    assert_approx_eq!(25f32.powf(-2f32), 0.0016f32);
    assert_approx_eq!(400f64.powf(0.5f64), 20f64);

    assert_approx_eq!(1f32.exp(), f32::consts::E);
    assert_approx_eq!(1f64.exp(), f64::consts::E);

    assert_approx_eq!(1f32.exp_m1(), f32::consts::E - 1.0);
    assert_approx_eq!(1f64.exp_m1(), f64::consts::E - 1.0);

    assert_approx_eq!(10f32.exp2(), 1024f32);
    assert_approx_eq!(50f64.exp2(), 1125899906842624f64);

    assert_approx_eq!(f32::consts::E.ln(), 1f32);
    assert_approx_eq!(1f64.ln(), 0f64);

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

    assert_approx_eq!(0f32.sin(), 0f32);
    assert_approx_eq!((f64::consts::PI / 2f64).sin(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_6.sin(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.sin().asin(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.sin().asin(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.sinh(), 1.1752012f32);
    assert_approx_eq!(1.0f64.sinh(), 1.1752012f64);
    assert_approx_eq!(2.0f32.asinh(), 1.443635475178810342493276740273105f32);
    assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);

    assert_approx_eq!(0f32.cos(), 1f32);
    assert_approx_eq!((f64::consts::PI * 2f64).cos(), 1f64);
    assert_approx_eq!(f32::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f64::consts::FRAC_PI_3.cos(), 0.5);
    assert_approx_eq!(f32::consts::FRAC_PI_4.cos().acos(), f32::consts::FRAC_PI_4);
    assert_approx_eq!(f64::consts::FRAC_PI_4.cos().acos(), f64::consts::FRAC_PI_4);

    assert_approx_eq!(1.0f32.cosh(), 1.54308f32);
    assert_approx_eq!(1.0f64.cosh(), 1.54308f64);
    assert_approx_eq!(2.0f32.acosh(), 1.31695789692481670862504634730796844f32);
    assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);

    assert_approx_eq!(1.0f32.tan(), 1.557408f32);
    assert_approx_eq!(1.0f64.tan(), 1.557408f64);
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
}

fn test_fast() {
    use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, frem_fast, fsub_fast};

    #[inline(never)]
    pub fn test_operations_f64(a: f64, b: f64) {
        // make sure they all map to the correct operation
        unsafe {
            assert_eq!(fadd_fast(a, b), a + b);
            assert_eq!(fsub_fast(a, b), a - b);
            assert_eq!(fmul_fast(a, b), a * b);
            assert_eq!(fdiv_fast(a, b), a / b);
            assert_eq!(frem_fast(a, b), a % b);
        }
    }

    #[inline(never)]
    pub fn test_operations_f32(a: f32, b: f32) {
        // make sure they all map to the correct operation
        unsafe {
            assert_eq!(fadd_fast(a, b), a + b);
            assert_eq!(fsub_fast(a, b), a - b);
            assert_eq!(fmul_fast(a, b), a * b);
            assert_eq!(fdiv_fast(a, b), a / b);
            assert_eq!(frem_fast(a, b), a % b);
        }
    }

    test_operations_f64(1., 2.);
    test_operations_f64(10., 5.);
    test_operations_f32(11., 2.);
    test_operations_f32(10., 15.);
}

fn test_algebraic() {
    use std::intrinsics::{
        fadd_algebraic, fdiv_algebraic, fmul_algebraic, frem_algebraic, fsub_algebraic,
    };

    #[inline(never)]
    pub fn test_operations_f64(a: f64, b: f64) {
        // make sure they all map to the correct operation
        assert_eq!(fadd_algebraic(a, b), a + b);
        assert_eq!(fsub_algebraic(a, b), a - b);
        assert_eq!(fmul_algebraic(a, b), a * b);
        assert_eq!(fdiv_algebraic(a, b), a / b);
        assert_eq!(frem_algebraic(a, b), a % b);
    }

    #[inline(never)]
    pub fn test_operations_f32(a: f32, b: f32) {
        // make sure they all map to the correct operation
        assert_eq!(fadd_algebraic(a, b), a + b);
        assert_eq!(fsub_algebraic(a, b), a - b);
        assert_eq!(fmul_algebraic(a, b), a * b);
        assert_eq!(fdiv_algebraic(a, b), a / b);
        assert_eq!(frem_algebraic(a, b), a % b);
    }

    test_operations_f64(1., 2.);
    test_operations_f64(10., 5.);
    test_operations_f32(11., 2.);
    test_operations_f32(10., 15.);
}
