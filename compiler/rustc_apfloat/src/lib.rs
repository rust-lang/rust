//! Port of LLVM's APFloat software floating-point implementation from the
//! following C++ sources (please update commit hash when backporting):
//! <https://github.com/llvm-mirror/llvm/tree/23efab2bbd424ed13495a420ad8641cb2c6c28f9>
//!
//! * `include/llvm/ADT/APFloat.h` -> `Float` and `FloatConvert` traits
//! * `lib/Support/APFloat.cpp` -> `ieee` and `ppc` modules
//! * `unittests/ADT/APFloatTest.cpp` -> `tests` directory
//!
//! The port contains no unsafe code, global state, or side-effects in general,
//! and the only allocations are in the conversion to/from decimal strings.
//!
//! Most of the API and the testcases are intact in some form or another,
//! with some ergonomic changes, such as idiomatic short names, returning
//! new values instead of mutating the receiver, and having separate method
//! variants that take a non-default rounding mode (with the suffix `_r`).
//! Comments have been preserved where possible, only slightly adapted.
//!
//! Instead of keeping a pointer to a configuration struct and inspecting it
//! dynamically on every operation, types (e.g., `ieee::Double`), traits
//! (e.g., `ieee::Semantics`) and associated constants are employed for
//! increased type safety and performance.
//!
//! On-heap bigints are replaced everywhere (except in decimal conversion),
//! with short arrays of `type Limb = u128` elements (instead of `u64`),
//! This allows fitting the largest supported significands in one integer
//! (`ieee::Quad` and `ppc::Fallback` use slightly less than 128 bits).
//! All of the functions in the `ieee::sig` module operate on slices.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![no_std]
#![forbid(unsafe_code)]
#![feature(nll)]

#[macro_use]
extern crate alloc;

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
use core::str::FromStr;

bitflags::bitflags! {
    /// IEEE-754R 7: Default exception handling.
    ///
    /// UNDERFLOW or OVERFLOW are always returned or-ed with INEXACT.
    #[must_use]
    pub struct Status: u8 {
        const OK = 0x00;
        const INVALID_OP = 0x01;
        const DIV_BY_ZERO = 0x02;
        const OVERFLOW = 0x04;
        const UNDERFLOW = 0x08;
        const INEXACT = 0x10;
    }
}

#[must_use]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct StatusAnd<T> {
    pub status: Status,
    pub value: T,
}

impl Status {
    pub fn and<T>(self, value: T) -> StatusAnd<T> {
        StatusAnd { status: self, value }
    }
}

impl<T> StatusAnd<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> StatusAnd<U> {
        StatusAnd { status: self.status, value: f(self.value) }
    }
}

#[macro_export]
macro_rules! unpack {
    ($status:ident|=, $e:expr) => {
        match $e {
            $crate::StatusAnd { status, value } => {
                $status |= status;
                value
            }
        }
    };
    ($status:ident=, $e:expr) => {
        match $e {
            $crate::StatusAnd { status, value } => {
                $status = status;
                value
            }
        }
    };
}

/// Category of internally-represented number.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Category {
    Infinity,
    NaN,
    Normal,
    Zero,
}

/// IEEE-754R 4.3: Rounding-direction attributes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Round {
    NearestTiesToEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    NearestTiesToAway,
}

impl Neg for Round {
    type Output = Round;
    fn neg(self) -> Round {
        match self {
            Round::TowardPositive => Round::TowardNegative,
            Round::TowardNegative => Round::TowardPositive,
            Round::NearestTiesToEven | Round::TowardZero | Round::NearestTiesToAway => self,
        }
    }
}

/// A signed type to represent a floating point number's unbiased exponent.
pub type ExpInt = i16;

// \c ilogb error results.
pub const IEK_INF: ExpInt = ExpInt::MAX;
pub const IEK_NAN: ExpInt = ExpInt::MIN;
pub const IEK_ZERO: ExpInt = ExpInt::MIN + 1;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ParseError(pub &'static str);

/// A self-contained host- and target-independent arbitrary-precision
/// floating-point software implementation.
///
/// `apfloat` uses significand bignum integer arithmetic as provided by functions
/// in the `ieee::sig`.
///
/// Written for clarity rather than speed, in particular with a view to use in
/// the front-end of a cross compiler so that target arithmetic can be correctly
/// performed on the host. Performance should nonetheless be reasonable,
/// particularly for its intended use. It may be useful as a base
/// implementation for a run-time library during development of a faster
/// target-specific one.
///
/// All 5 rounding modes in the IEEE-754R draft are handled correctly for all
/// implemented operations. Currently implemented operations are add, subtract,
/// multiply, divide, fused-multiply-add, conversion-to-float,
/// conversion-to-integer and conversion-from-integer. New rounding modes
/// (e.g., away from zero) can be added with three or four lines of code.
///
/// Four formats are built-in: IEEE single precision, double precision,
/// quadruple precision, and x87 80-bit extended double (when operating with
/// full extended precision). Adding a new format that obeys IEEE semantics
/// only requires adding two lines of code: a declaration and definition of the
/// format.
///
/// All operations return the status of that operation as an exception bit-mask,
/// so multiple operations can be done consecutively with their results or-ed
/// together. The returned status can be useful for compiler diagnostics; e.g.,
/// inexact, underflow and overflow can be easily diagnosed on constant folding,
/// and compiler optimizers can determine what exceptions would be raised by
/// folding operations and optimize, or perhaps not optimize, accordingly.
///
/// At present, underflow tininess is detected after rounding; it should be
/// straight forward to add support for the before-rounding case too.
///
/// The library reads hexadecimal floating point numbers as per C99, and
/// correctly rounds if necessary according to the specified rounding mode.
/// Syntax is required to have been validated by the caller.
///
/// It also reads decimal floating point numbers and correctly rounds according
/// to the specified rounding mode.
///
/// Non-zero finite numbers are represented internally as a sign bit, a 16-bit
/// signed exponent, and the significand as an array of integer limbs. After
/// normalization of a number of precision P the exponent is within the range of
/// the format, and if the number is not denormal the P-th bit of the
/// significand is set as an explicit integer bit. For denormals the most
/// significant bit is shifted right so that the exponent is maintained at the
/// format's minimum, so that the smallest denormal has just the least
/// significant bit of the significand set. The sign of zeros and infinities
/// is significant; the exponent and significand of such numbers is not stored,
/// but has a known implicit (deterministic) value: 0 for the significands, 0
/// for zero exponent, all 1 bits for infinity exponent. For NaNs the sign and
/// significand are deterministic, although not really meaningful, and preserved
/// in non-conversion operations. The exponent is implicitly all 1 bits.
///
/// `apfloat` does not provide any exception handling beyond default exception
/// handling. We represent Signaling NaNs via IEEE-754R 2008 6.2.1 should clause
/// by encoding Signaling NaNs with the first bit of its trailing significand
/// as 0.
///
/// Future work
/// ===========
///
/// Some features that may or may not be worth adding:
///
/// Optional ability to detect underflow tininess before rounding.
///
/// New formats: x87 in single and double precision mode (IEEE apart from
/// extended exponent range) (hard).
///
/// New operations: sqrt, nexttoward.
///
pub trait Float:
    Copy
    + Default
    + FromStr<Err = ParseError>
    + PartialOrd
    + fmt::Display
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Add<Output = StatusAnd<Self>>
    + Sub<Output = StatusAnd<Self>>
    + Mul<Output = StatusAnd<Self>>
    + Div<Output = StatusAnd<Self>>
    + Rem<Output = StatusAnd<Self>>
{
    /// Total number of bits in the in-memory format.
    const BITS: usize;

    /// Number of bits in the significand. This includes the integer bit.
    const PRECISION: usize;

    /// The largest E such that 2<sup>E</sup> is representable; this matches the
    /// definition of IEEE 754.
    const MAX_EXP: ExpInt;

    /// The smallest E such that 2<sup>E</sup> is a normalized number; this
    /// matches the definition of IEEE 754.
    const MIN_EXP: ExpInt;

    /// Positive Zero.
    const ZERO: Self;

    /// Positive Infinity.
    const INFINITY: Self;

    /// NaN (Not a Number).
    // FIXME(eddyb) provide a default when qnan becomes const fn.
    const NAN: Self;

    /// Factory for QNaN values.
    // FIXME(eddyb) should be const fn.
    fn qnan(payload: Option<u128>) -> Self;

    /// Factory for SNaN values.
    // FIXME(eddyb) should be const fn.
    fn snan(payload: Option<u128>) -> Self;

    /// Largest finite number.
    // FIXME(eddyb) should be const (but FloatPair::largest is nontrivial).
    fn largest() -> Self;

    /// Smallest (by magnitude) finite number.
    /// Might be denormalized, which implies a relative loss of precision.
    const SMALLEST: Self;

    /// Smallest (by magnitude) normalized finite number.
    // FIXME(eddyb) should be const (but FloatPair::smallest_normalized is nontrivial).
    fn smallest_normalized() -> Self;

    // Arithmetic

    fn add_r(self, rhs: Self, round: Round) -> StatusAnd<Self>;
    fn sub_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        self.add_r(-rhs, round)
    }
    fn mul_r(self, rhs: Self, round: Round) -> StatusAnd<Self>;
    fn mul_add_r(self, multiplicand: Self, addend: Self, round: Round) -> StatusAnd<Self>;
    fn mul_add(self, multiplicand: Self, addend: Self) -> StatusAnd<Self> {
        self.mul_add_r(multiplicand, addend, Round::NearestTiesToEven)
    }
    fn div_r(self, rhs: Self, round: Round) -> StatusAnd<Self>;
    /// IEEE remainder.
    // This is not currently correct in all cases.
    fn ieee_rem(self, rhs: Self) -> StatusAnd<Self> {
        let mut v = self;

        let status;
        v = unpack!(status=, v / rhs);
        if status == Status::DIV_BY_ZERO {
            return status.and(self);
        }

        assert!(Self::PRECISION < 128);

        let status;
        let x = unpack!(status=, v.to_i128_r(128, Round::NearestTiesToEven, &mut false));
        if status == Status::INVALID_OP {
            return status.and(self);
        }

        let status;
        let mut v = unpack!(status=, Self::from_i128(x));
        assert_eq!(status, Status::OK); // should always work

        let status;
        v = unpack!(status=, v * rhs);
        assert_eq!(status - Status::INEXACT, Status::OK); // should not overflow or underflow

        let status;
        v = unpack!(status=, self - v);
        assert_eq!(status - Status::INEXACT, Status::OK); // likewise

        if v.is_zero() {
            status.and(v.copy_sign(self)) // IEEE754 requires this
        } else {
            status.and(v)
        }
    }
    /// C fmod, or llvm frem.
    fn c_fmod(self, rhs: Self) -> StatusAnd<Self>;
    fn round_to_integral(self, round: Round) -> StatusAnd<Self>;

    /// IEEE-754R 2008 5.3.1: nextUp.
    fn next_up(self) -> StatusAnd<Self>;

    /// IEEE-754R 2008 5.3.1: nextDown.
    ///
    /// *NOTE* since nextDown(x) = -nextUp(-x), we only implement nextUp with
    /// appropriate sign switching before/after the computation.
    fn next_down(self) -> StatusAnd<Self> {
        (-self).next_up().map(|r| -r)
    }

    fn abs(self) -> Self {
        if self.is_negative() { -self } else { self }
    }
    fn copy_sign(self, rhs: Self) -> Self {
        if self.is_negative() != rhs.is_negative() { -self } else { self }
    }

    // Conversions
    fn from_bits(input: u128) -> Self;
    fn from_i128_r(input: i128, round: Round) -> StatusAnd<Self> {
        if input < 0 {
            Self::from_u128_r(input.wrapping_neg() as u128, -round).map(|r| -r)
        } else {
            Self::from_u128_r(input as u128, round)
        }
    }
    fn from_i128(input: i128) -> StatusAnd<Self> {
        Self::from_i128_r(input, Round::NearestTiesToEven)
    }
    fn from_u128_r(input: u128, round: Round) -> StatusAnd<Self>;
    fn from_u128(input: u128) -> StatusAnd<Self> {
        Self::from_u128_r(input, Round::NearestTiesToEven)
    }
    fn from_str_r(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError>;
    fn to_bits(self) -> u128;

    /// Converts a floating point number to an integer according to the
    /// rounding mode. In case of an invalid operation exception,
    /// deterministic values are returned, namely zero for NaNs and the
    /// minimal or maximal value respectively for underflow or overflow.
    /// If the rounded value is in range but the floating point number is
    /// not the exact integer, the C standard doesn't require an inexact
    /// exception to be raised. IEEE-854 does require it so we do that.
    ///
    /// Note that for conversions to integer type the C standard requires
    /// round-to-zero to always be used.
    ///
    /// The *is_exact output tells whether the result is exact, in the sense
    /// that converting it back to the original floating point type produces
    /// the original value. This is almost equivalent to `result == Status::OK`,
    /// except for negative zeroes.
    fn to_i128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<i128> {
        let status;
        if self.is_negative() {
            if self.is_zero() {
                // Negative zero can't be represented as an int.
                *is_exact = false;
            }
            let r = unpack!(status=, (-self).to_u128_r(width, -round, is_exact));

            // Check for values that don't fit in the signed integer.
            if r > (1 << (width - 1)) {
                // Return the most negative integer for the given width.
                *is_exact = false;
                Status::INVALID_OP.and(-1 << (width - 1))
            } else {
                status.and(r.wrapping_neg() as i128)
            }
        } else {
            // Positive case is simpler, can pretend it's a smaller unsigned
            // integer, and `to_u128` will take care of all the edge cases.
            self.to_u128_r(width - 1, round, is_exact).map(|r| r as i128)
        }
    }
    fn to_i128(self, width: usize) -> StatusAnd<i128> {
        self.to_i128_r(width, Round::TowardZero, &mut true)
    }
    fn to_u128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<u128>;
    fn to_u128(self, width: usize) -> StatusAnd<u128> {
        self.to_u128_r(width, Round::TowardZero, &mut true)
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering;

    /// Bitwise comparison for equality (QNaNs compare equal, 0!=-0).
    fn bitwise_eq(self, rhs: Self) -> bool;

    // IEEE-754R 5.7.2 General operations.

    /// Implements IEEE minNum semantics. Returns the smaller of the 2 arguments if
    /// both are not NaN. If either argument is a NaN, returns the other argument.
    fn min(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if other.partial_cmp(&self) == Some(Ordering::Less) {
            other
        } else {
            self
        }
    }

    /// Implements IEEE maxNum semantics. Returns the larger of the 2 arguments if
    /// both are not NaN. If either argument is a NaN, returns the other argument.
    fn max(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if self.partial_cmp(&other) == Some(Ordering::Less) {
            other
        } else {
            self
        }
    }

    /// IEEE-754R isSignMinus: Returns whether the current value is
    /// negative.
    ///
    /// This applies to zeros and NaNs as well.
    fn is_negative(self) -> bool;

    /// IEEE-754R isNormal: Returns whether the current value is normal.
    ///
    /// This implies that the current value of the float is not zero, subnormal,
    /// infinite, or NaN following the definition of normality from IEEE-754R.
    fn is_normal(self) -> bool {
        !self.is_denormal() && self.is_finite_non_zero()
    }

    /// Returns `true` if the current value is zero, subnormal, or
    /// normal.
    ///
    /// This means that the value is not infinite or NaN.
    fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Returns `true` if the float is plus or minus zero.
    fn is_zero(self) -> bool {
        self.category() == Category::Zero
    }

    /// IEEE-754R isSubnormal(): Returns whether the float is a
    /// denormal.
    fn is_denormal(self) -> bool;

    /// IEEE-754R isInfinite(): Returns whether the float is infinity.
    fn is_infinite(self) -> bool {
        self.category() == Category::Infinity
    }

    /// Returns `true` if the float is a quiet or signaling NaN.
    fn is_nan(self) -> bool {
        self.category() == Category::NaN
    }

    /// Returns `true` if the float is a signaling NaN.
    fn is_signaling(self) -> bool;

    // Simple Queries

    fn category(self) -> Category;
    fn is_non_zero(self) -> bool {
        !self.is_zero()
    }
    fn is_finite_non_zero(self) -> bool {
        self.is_finite() && !self.is_zero()
    }
    fn is_pos_zero(self) -> bool {
        self.is_zero() && !self.is_negative()
    }
    fn is_neg_zero(self) -> bool {
        self.is_zero() && self.is_negative()
    }

    /// Returns `true` if the number has the smallest possible non-zero
    /// magnitude in the current semantics.
    fn is_smallest(self) -> bool {
        Self::SMALLEST.copy_sign(self).bitwise_eq(self)
    }

    /// Returns `true` if the number has the largest possible finite
    /// magnitude in the current semantics.
    fn is_largest(self) -> bool {
        Self::largest().copy_sign(self).bitwise_eq(self)
    }

    /// Returns `true` if the number is an exact integer.
    fn is_integer(self) -> bool {
        // This could be made more efficient; I'm going for obviously correct.
        if !self.is_finite() {
            return false;
        }
        self.round_to_integral(Round::TowardZero).value.bitwise_eq(self)
    }

    /// If this value has an exact multiplicative inverse, return it.
    fn get_exact_inverse(self) -> Option<Self>;

    /// Returns the exponent of the internal representation of the Float.
    ///
    /// Because the radix of Float is 2, this is equivalent to floor(log2(x)).
    /// For special Float values, this returns special error codes:
    ///
    ///   NaN -> \c IEK_NAN
    ///   0   -> \c IEK_ZERO
    ///   Inf -> \c IEK_INF
    ///
    fn ilogb(self) -> ExpInt;

    /// Returns: self * 2<sup>exp</sup> for integral exponents.
    /// Equivalent to C standard library function `ldexp`.
    fn scalbn_r(self, exp: ExpInt, round: Round) -> Self;
    fn scalbn(self, exp: ExpInt) -> Self {
        self.scalbn_r(exp, Round::NearestTiesToEven)
    }

    /// Equivalent to C standard library function with the same name.
    ///
    /// While the C standard says exp is an unspecified value for infinity and nan,
    /// this returns INT_MAX for infinities, and INT_MIN for NaNs (see `ilogb`).
    fn frexp_r(self, exp: &mut ExpInt, round: Round) -> Self;
    fn frexp(self, exp: &mut ExpInt) -> Self {
        self.frexp_r(exp, Round::NearestTiesToEven)
    }
}

pub trait FloatConvert<T: Float>: Float {
    /// Converts a value of one floating point type to another.
    /// The return value corresponds to the IEEE754 exceptions. *loses_info
    /// records whether the transformation lost information, i.e., whether
    /// converting the result back to the original type will produce the
    /// original value (this is almost the same as return `value == Status::OK`,
    /// but there are edge cases where this is not so).
    fn convert_r(self, round: Round, loses_info: &mut bool) -> StatusAnd<T>;
    fn convert(self, loses_info: &mut bool) -> StatusAnd<T> {
        self.convert_r(Round::NearestTiesToEven, loses_info)
    }
}

macro_rules! float_common_impls {
    ($ty:ident<$t:tt>) => {
        impl<$t> Default for $ty<$t>
        where
            Self: Float,
        {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl<$t> ::core::str::FromStr for $ty<$t>
        where
            Self: Float,
        {
            type Err = ParseError;
            fn from_str(s: &str) -> Result<Self, ParseError> {
                Self::from_str_r(s, Round::NearestTiesToEven).map(|x| x.value)
            }
        }

        // Rounding ties to the nearest even, by default.

        impl<$t> ::core::ops::Add for $ty<$t>
        where
            Self: Float,
        {
            type Output = StatusAnd<Self>;
            fn add(self, rhs: Self) -> StatusAnd<Self> {
                self.add_r(rhs, Round::NearestTiesToEven)
            }
        }

        impl<$t> ::core::ops::Sub for $ty<$t>
        where
            Self: Float,
        {
            type Output = StatusAnd<Self>;
            fn sub(self, rhs: Self) -> StatusAnd<Self> {
                self.sub_r(rhs, Round::NearestTiesToEven)
            }
        }

        impl<$t> ::core::ops::Mul for $ty<$t>
        where
            Self: Float,
        {
            type Output = StatusAnd<Self>;
            fn mul(self, rhs: Self) -> StatusAnd<Self> {
                self.mul_r(rhs, Round::NearestTiesToEven)
            }
        }

        impl<$t> ::core::ops::Div for $ty<$t>
        where
            Self: Float,
        {
            type Output = StatusAnd<Self>;
            fn div(self, rhs: Self) -> StatusAnd<Self> {
                self.div_r(rhs, Round::NearestTiesToEven)
            }
        }

        impl<$t> ::core::ops::Rem for $ty<$t>
        where
            Self: Float,
        {
            type Output = StatusAnd<Self>;
            fn rem(self, rhs: Self) -> StatusAnd<Self> {
                self.c_fmod(rhs)
            }
        }

        impl<$t> ::core::ops::AddAssign for $ty<$t>
        where
            Self: Float,
        {
            fn add_assign(&mut self, rhs: Self) {
                *self = (*self + rhs).value;
            }
        }

        impl<$t> ::core::ops::SubAssign for $ty<$t>
        where
            Self: Float,
        {
            fn sub_assign(&mut self, rhs: Self) {
                *self = (*self - rhs).value;
            }
        }

        impl<$t> ::core::ops::MulAssign for $ty<$t>
        where
            Self: Float,
        {
            fn mul_assign(&mut self, rhs: Self) {
                *self = (*self * rhs).value;
            }
        }

        impl<$t> ::core::ops::DivAssign for $ty<$t>
        where
            Self: Float,
        {
            fn div_assign(&mut self, rhs: Self) {
                *self = (*self / rhs).value;
            }
        }

        impl<$t> ::core::ops::RemAssign for $ty<$t>
        where
            Self: Float,
        {
            fn rem_assign(&mut self, rhs: Self) {
                *self = (*self % rhs).value;
            }
        }
    };
}

pub mod ieee;
pub mod ppc;
