//! Helper trait for generic float types.

use core::f64;

use crate::fmt::{Debug, LowerExp};
use crate::num::FpCategory;
use crate::ops::{self, Add, Div, Mul, Neg};

/// Lossy `as` casting between two types.
pub trait CastInto<T: Copy>: Copy {
    fn cast(self) -> T;
}

/// Collection of traits that allow us to be generic over integer size.
pub trait Integer:
    Sized
    + Clone
    + Copy
    + Debug
    + ops::Shr<u32, Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + PartialEq
    + CastInto<i16>
{
    const ZERO: Self;
    const ONE: Self;
}

macro_rules! int {
    ($($ty:ty),+) => {
        $(
            impl CastInto<i16> for $ty {
                fn cast(self) -> i16 {
                    self as i16
                }
            }

            impl Integer for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
            }
        )+
    }
}

int!(u16, u32, u64);

/// A helper trait to avoid duplicating basically all the conversion code for IEEE floats.
///
/// See the parent module's doc comment for why this is necessary.
///
/// Should **never ever** be implemented for other types or be used outside the `dec2flt` module.
#[doc(hidden)]
pub trait RawFloat:
    Sized
    + Div<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Add<Output = Self>
    + LowerExp
    + PartialEq
    + PartialOrd
    + Default
    + Clone
    + Copy
    + Debug
{
    /// The unsigned integer with the same size as the float
    type Int: Integer + Into<u64>;

    /* general constants */

    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const NEG_NAN: Self;

    /// Bit width of the float
    const BITS: u32;

    /// The number of bits in the significand, *including* the hidden bit.
    const SIG_TOTAL_BITS: u32;

    const EXP_MASK: Self::Int;
    const SIG_MASK: Self::Int;

    /// The number of bits in the significand, *excluding* the hidden bit.
    const SIG_BITS: u32 = Self::SIG_TOTAL_BITS - 1;

    /// Number of bits in the exponent.
    const EXP_BITS: u32 = Self::BITS - Self::SIG_BITS - 1;

    /// The saturated (maximum bitpattern) value of the exponent, i.e. the infinite
    /// representation.
    ///
    /// This shifted fully right, use `EXP_MASK` for the shifted value.
    const EXP_SAT: u32 = (1 << Self::EXP_BITS) - 1;

    /// Signed version of `EXP_SAT` since we convert a lot.
    const INFINITE_POWER: i32 = Self::EXP_SAT as i32;

    /// The exponent bias value. This is also the maximum value of the exponent.
    const EXP_BIAS: u32 = Self::EXP_SAT >> 1;

    /// Minimum exponent value of normal values.
    const EXP_MIN: i32 = -(Self::EXP_BIAS as i32 - 1);

    /// Round-to-even only happens for negative values of q
    /// when q ≥ −4 in the 64-bit case and when q ≥ −17 in
    /// the 32-bitcase.
    ///
    /// When q ≥ 0,we have that 5^q ≤ 2m+1. In the 64-bit case,we
    /// have 5^q ≤ 2m+1 ≤ 2^54 or q ≤ 23. In the 32-bit case,we have
    /// 5^q ≤ 2m+1 ≤ 2^25 or q ≤ 10.
    ///
    /// When q < 0, we have w ≥ (2m+1)×5^−q. We must have that w < 2^64
    /// so (2m+1)×5^−q < 2^64. We have that 2m+1 > 2^53 (64-bit case)
    /// or 2m+1 > 2^24 (32-bit case). Hence,we must have 2^53×5^−q < 2^64
    /// (64-bit) and 2^24×5^−q < 2^64 (32-bit). Hence we have 5^−q < 2^11
    /// or q ≥ −4 (64-bit case) and 5^−q < 2^40 or q ≥ −17 (32-bitcase).
    ///
    /// Thus we have that we only need to round ties to even when
    /// we have that q ∈ [−4,23](in the 64-bit case) or q∈[−17,10]
    /// (in the 32-bit case). In both cases,the power of five(5^|q|)
    /// fits in a 64-bit word.
    const MIN_EXPONENT_ROUND_TO_EVEN: i32;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32;

    /* limits related to Fast pathing */

    /// Largest decimal exponent for a non-infinite value.
    ///
    /// This is the max exponent in binary converted to the max exponent in decimal. Allows fast
    /// pathing anything larger than `10^LARGEST_POWER_OF_TEN`, which will round to infinity.
    const LARGEST_POWER_OF_TEN: i32 = {
        let largest_pow2 = Self::EXP_BIAS + 1;
        pow2_to_pow10(largest_pow2 as i64) as i32
    };

    /// Smallest decimal exponent for a non-zero value. This allows for fast pathing anything
    /// smaller than `10^SMALLEST_POWER_OF_TEN`, which will round to zero.
    ///
    /// The smallest power of ten is represented by `⌊log10(2^-n / (2^64 - 1))⌋`, where `n` is
    /// the smallest power of two. The `2^64 - 1)` denomenator comes from the number of values
    /// that are representable by the intermediate storage format. I don't actually know _why_
    /// the storage format is relevant here.
    ///
    /// The values may be calculated using the formula. Unfortunately we cannot calculate them at
    /// compile time since intermediates exceed the range of an `f64`.
    const SMALLEST_POWER_OF_TEN: i32;

    /// Maximum exponent for a fast path case, or `⌊(SIG_BITS+1)/log2(5)⌋`
    // assuming FLT_EVAL_METHOD = 0
    const MAX_EXPONENT_FAST_PATH: i64 = {
        let log2_5 = f64::consts::LOG2_10 - 1.0;
        (Self::SIG_TOTAL_BITS as f64 / log2_5) as i64
    };

    /// Minimum exponent for a fast path case, or `-⌊(SIG_BITS+1)/log2(5)⌋`
    const MIN_EXPONENT_FAST_PATH: i64 = -Self::MAX_EXPONENT_FAST_PATH;

    /// Maximum exponent that can be represented for a disguised-fast path case.
    /// This is `MAX_EXPONENT_FAST_PATH + ⌊(SIG_BITS+1)/log2(10)⌋`
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i64 =
        Self::MAX_EXPONENT_FAST_PATH + (Self::SIG_TOTAL_BITS as f64 / f64::consts::LOG2_10) as i64;

    /// Maximum mantissa for the fast-path (`1 << 53` for f64).
    const MAX_MANTISSA_FAST_PATH: u64 = 1 << Self::SIG_TOTAL_BITS;

    /// Converts integer into float through an as cast.
    /// This is only called in the fast-path algorithm, and therefore
    /// will not lose precision, since the value will always have
    /// only if the value is <= Self::MAX_MANTISSA_FAST_PATH.
    fn from_u64(v: u64) -> Self;

    /// Performs a raw transmutation from an integer.
    fn from_u64_bits(v: u64) -> Self;

    /// Gets a small power-of-ten for fast-path multiplication.
    fn pow10_fast_path(exponent: usize) -> Self;

    /// Returns the category that this number falls into.
    fn classify(self) -> FpCategory;

    /// Transmute to the integer representation
    fn to_bits(self) -> Self::Int;

    /// Returns the mantissa, exponent and sign as integers.
    ///
    /// This returns `(m, p, s)` such that `s * m * 2^p` represents the original float. For 0, the
    /// exponent will be `-(EXP_BIAS + SIG_BITS)`, which is the minimum subnormal power. For
    /// infinity or NaN, the exponent will be `EXP_SAT - EXP_BIAS - SIG_BITS`.
    ///
    /// If subnormal, the mantissa will be shifted one bit to the left. Otherwise, it is returned
    /// with the explicit bit set but otherwise unshifted
    ///
    /// `s` is only ever +/-1.
    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> (Self::BITS - 1) == Self::Int::ZERO { 1 } else { -1 };
        let mut exponent: i16 = ((bits & Self::EXP_MASK) >> Self::SIG_BITS).cast();
        let mantissa = if exponent == 0 {
            (bits & Self::SIG_MASK) << 1
        } else {
            (bits & Self::SIG_MASK) | (Self::Int::ONE << Self::SIG_BITS)
        };
        // Exponent bias + mantissa shift
        exponent -= (Self::EXP_BIAS + Self::SIG_BITS) as i16;
        (mantissa.into(), exponent, sign)
    }
}

/// Solve for `b` in `10^b = 2^a`
const fn pow2_to_pow10(a: i64) -> i64 {
    let res = (a as f64) / f64::consts::LOG2_10;
    res as i64
}

#[cfg(target_has_reliable_f16)]
impl RawFloat for f16 {
    type Int = u16;

    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self = Self::NEG_INFINITY;
    const NAN: Self = Self::NAN;
    const NEG_NAN: Self = -Self::NAN;

    const BITS: u32 = 16;
    const SIG_TOTAL_BITS: u32 = Self::MANTISSA_DIGITS;
    const EXP_MASK: Self::Int = Self::EXP_MASK;
    const SIG_MASK: Self::Int = Self::MAN_MASK;

    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -22;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 5;
    const SMALLEST_POWER_OF_TEN: i32 = -27;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        Self::from_bits((v & 0xFFFF) as u16)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        #[allow(clippy::use_self)]
        const TABLE: [f16; 8] = [1e0, 1e1, 1e2, 1e3, 1e4, 0.0, 0.0, 0.];
        TABLE[exponent & 7]
    }

    fn to_bits(self) -> Self::Int {
        self.to_bits()
    }

    fn classify(self) -> FpCategory {
        self.classify()
    }
}

impl RawFloat for f32 {
    type Int = u32;

    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const NEG_NAN: Self = -f32::NAN;

    const BITS: u32 = 32;
    const SIG_TOTAL_BITS: u32 = Self::MANTISSA_DIGITS;
    const EXP_MASK: Self::Int = Self::EXP_MASK;
    const SIG_MASK: Self::Int = Self::MAN_MASK;

    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -17;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 10;
    const SMALLEST_POWER_OF_TEN: i32 = -65;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        f32::from_bits((v & 0xFFFFFFFF) as u32)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        #[allow(clippy::use_self)]
        const TABLE: [f32; 16] =
            [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 0., 0., 0., 0., 0.];
        TABLE[exponent & 15]
    }

    fn to_bits(self) -> Self::Int {
        self.to_bits()
    }

    fn classify(self) -> FpCategory {
        self.classify()
    }
}

impl RawFloat for f64 {
    type Int = u64;

    const INFINITY: Self = Self::INFINITY;
    const NEG_INFINITY: Self = Self::NEG_INFINITY;
    const NAN: Self = Self::NAN;
    const NEG_NAN: Self = -Self::NAN;

    const BITS: u32 = 64;
    const SIG_TOTAL_BITS: u32 = Self::MANTISSA_DIGITS;
    const EXP_MASK: Self::Int = Self::EXP_MASK;
    const SIG_MASK: Self::Int = Self::MAN_MASK;

    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -4;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 23;
    const SMALLEST_POWER_OF_TEN: i32 = -342;

    #[inline]
    fn from_u64(v: u64) -> Self {
        debug_assert!(v <= Self::MAX_MANTISSA_FAST_PATH);
        v as _
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        f64::from_bits(v)
    }

    fn pow10_fast_path(exponent: usize) -> Self {
        const TABLE: [f64; 32] = [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
            1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        TABLE[exponent & 31]
    }

    fn to_bits(self) -> Self::Int {
        self.to_bits()
    }

    fn classify(self) -> FpCategory {
        self.classify()
    }
}
