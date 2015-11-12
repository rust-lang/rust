// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits and functions for the built-in numeric types.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

use self::wrapping::OverflowingOps;

use char::CharExt;
use cmp::{Eq, PartialOrd};
use convert::From;
use fmt;
use intrinsics;
use marker::{Copy, Sized};
use mem::size_of;
use option::Option::{self, Some, None};
use result::Result::{self, Ok, Err};
use str::{FromStr, StrExt};
use slice::SliceExt;

/// Provides intentionally-wrapped arithmetic on `T`.
///
/// Operations like `+` on `u32` values is intended to never overflow,
/// and in some debug configurations overflow is detected and results
/// in a panic. While most arithmetic falls into this category, some
/// code explicitly expects and relies upon modular arithmetic (e.g.,
/// hashing).
///
/// Wrapping arithmetic can be achieved either through methods like
/// `wrapping_add`, or through the `Wrapping<T>` type, which says that
/// all standard arithmetic operations on the underlying value are
/// intended to have wrapping semantics.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Default)]
pub struct Wrapping<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

pub mod wrapping;

// All these modules are technically private and only exposed for libcoretest:
pub mod flt2dec;
pub mod dec2flt;
pub mod bignum;
pub mod diy_float;

/// Types that have a "zero" value.
///
/// This trait is intended for use in conjunction with `Add`, as an identity:
/// `x + T::zero() == x`.
#[unstable(feature = "zero_one",
           reason = "unsure of placement, wants to use associated constants",
           issue = "27739")]
pub trait Zero: Sized {
    /// The "zero" (usually, additive identity) for this type.
    fn zero() -> Self;
}

/// Types that have a "one" value.
///
/// This trait is intended for use in conjunction with `Mul`, as an identity:
/// `x * T::one() == x`.
#[unstable(feature = "zero_one",
           reason = "unsure of placement, wants to use associated constants",
           issue = "27739")]
pub trait One: Sized {
    /// The "one" (usually, multiplicative identity) for this type.
    fn one() -> Self;
}

macro_rules! zero_one_impl {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            #[inline]
            fn zero() -> Self { 0 }
        }
        impl One for $t {
            #[inline]
            fn one() -> Self { 1 }
        }
    )*)
}
zero_one_impl! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

macro_rules! zero_one_impl_float {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            #[inline]
            fn zero() -> Self { 0.0 }
        }
        impl One for $t {
            #[inline]
            fn one() -> Self { 1.0 }
        }
    )*)
}
zero_one_impl_float! { f32 f64 }

// Just for stage0; a byte swap on a byte is a no-op
// Delete this once it becomes unused
#[cfg(stage0)]
unsafe fn bswap8(x: u8) -> u8 { x }

macro_rules! checked_op {
    ($U:ty, $op:path, $x:expr, $y:expr) => {{
        let (result, overflowed) = unsafe { $op($x as $U, $y as $U) };
        if overflowed { None } else { Some(result as Self) }
    }}
}

// `Int` + `SignedInt` implemented for signed integers
macro_rules! int_impl {
    ($ActualT:ty, $UnsignedT:ty, $BITS:expr,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn min_value() -> Self {
            (-1 as Self) << ($BITS - 1)
        }

        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn max_value() -> Self {
            let min = Self::min_value(); !min
        }

        /// Converts a string slice in a given base to an integer.
        ///
        /// Leading and trailing whitespace represent an error.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(u32::from_str_radix("A", 16), Ok(10));
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
            from_str_radix(src, radix)
        }

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b01001100u8;
        ///
        /// assert_eq!(n.count_ones(), 3);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn count_ones(self) -> u32 { (self as $UnsignedT).count_ones() }

        /// Returns the number of zeros in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b01001100u8;
        ///
        /// assert_eq!(n.count_zeros(), 5);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn count_zeros(self) -> u32 {
            (!self).count_ones()
        }

        /// Returns the number of leading zeros in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b0101000u16;
        ///
        /// assert_eq!(n.leading_zeros(), 10);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn leading_zeros(self) -> u32 {
            (self as $UnsignedT).leading_zeros()
        }

        /// Returns the number of trailing zeros in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b0101000u16;
        ///
        /// assert_eq!(n.trailing_zeros(), 3);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn trailing_zeros(self) -> u32 {
            (self as $UnsignedT).trailing_zeros()
        }

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0x3456789ABCDEF012u64;
        ///
        /// assert_eq!(n.rotate_left(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_left(self, n: u32) -> Self {
            (self as $UnsignedT).rotate_left(n) as Self
        }

        /// Shifts the bits to the right by a specified amount, `n`,
        /// wrapping the truncated bits to the beginning of the resulting
        /// integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xDEF0123456789ABCu64;
        ///
        /// assert_eq!(n.rotate_right(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_right(self, n: u32) -> Self {
            (self as $UnsignedT).rotate_right(n) as Self
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xEFCDAB8967452301u64;
        ///
        /// assert_eq!(n.swap_bytes(), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn swap_bytes(self) -> Self {
            (self as $UnsignedT).swap_bytes() as Self
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(u64::from_be(n), n)
        /// } else {
        ///     assert_eq!(u64::from_be(n), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn from_be(x: Self) -> Self {
            if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
        }

        /// Converts an integer from little endian to the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(u64::from_le(n), n)
        /// } else {
        ///     assert_eq!(u64::from_le(n), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn from_le(x: Self) -> Self {
            if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
        }

        /// Converts `self` to big endian from the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(n.to_be(), n)
        /// } else {
        ///     assert_eq!(n.to_be(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn to_be(self) -> Self { // or not to be?
            if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
        }

        /// Converts `self` to little endian from the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(n.to_le(), n)
        /// } else {
        ///     assert_eq!(n.to_le(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn to_le(self) -> Self {
            if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
        }

        /// Checked integer addition. Computes `self + other`, returning `None`
        /// if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(5u16.checked_add(65530), Some(65535));
        /// assert_eq!(6u16.checked_add(65530), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_add(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $add_with_overflow, self, other)
        }

        /// Checked integer subtraction. Computes `self - other`, returning
        /// `None` if underflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-127i8).checked_sub(1), Some(-128));
        /// assert_eq!((-128i8).checked_sub(1), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_sub(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $sub_with_overflow, self, other)
        }

        /// Checked integer multiplication. Computes `self * other`, returning
        /// `None` if underflow or overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(5u8.checked_mul(51), Some(255));
        /// assert_eq!(5u8.checked_mul(52), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_mul(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $mul_with_overflow, self, other)
        }

        /// Checked integer division. Computes `self / other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-127i8).checked_div(-1), Some(127));
        /// assert_eq!((-128i8).checked_div(-1), None);
        /// assert_eq!((1i8).checked_div(0), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_div(self, other: Self) -> Option<Self> {
            match other {
                0    => None,
               -1 if self == Self::min_value()
                     => None,
               other => Some(self / other),
            }
        }

        /// Saturating integer addition. Computes `self + other`, saturating at
        /// the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.saturating_add(1), 101);
        /// assert_eq!(100i8.saturating_add(127), 127);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_add(self, other: Self) -> Self {
            match self.checked_add(other) {
                Some(x)                       => x,
                None if other >= Self::zero() => Self::max_value(),
                None => Self::min_value(),
            }
        }

        /// Saturating integer subtraction. Computes `self - other`, saturating
        /// at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.saturating_sub(127), -27);
        /// assert_eq!((-100i8).saturating_sub(127), -128);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_sub(self, other: Self) -> Self {
            match self.checked_sub(other) {
                Some(x)                      => x,
                None if other >= Self::zero() => Self::min_value(),
                None => Self::max_value(),
            }
        }

        /// Wrapping (modular) addition. Computes `self + other`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_add(27), 127);
        /// assert_eq!(100i8.wrapping_add(127), -29);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_add(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_add(self, rhs)
            }
        }

        /// Wrapping (modular) subtraction. Computes `self - other`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0i8.wrapping_sub(127), -127);
        /// assert_eq!((-2i8).wrapping_sub(127), 127);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_sub(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_sub(self, rhs)
            }
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// other`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(10i8.wrapping_mul(12), 120);
        /// assert_eq!(11i8.wrapping_mul(12), -124);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_mul(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_mul(self, rhs)
            }
        }

        /// Wrapping (modular) division. Computes `self / other`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// divides `MIN / -1` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is equivalent
        /// to `-MIN`, a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN`
        /// itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.wrapping_div(10), 10);
        /// assert_eq!((-128i8).wrapping_div(-1), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_div(self, rhs: Self) -> Self {
            self.overflowing_div(rhs).0
        }

        /// Wrapping (modular) remainder. Computes `self % other`,
        /// wrapping around at the boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically;
        /// implementation artifacts make `x % y` invalid for `MIN /
        /// -1` on a signed type (where `MIN` is the negative
        /// minimal value). In such a case, this function returns `0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_rem(10), 0);
        /// assert_eq!((-128i8).wrapping_rem(-1), 0);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_rem(self, rhs: Self) -> Self {
            self.overflowing_rem(rhs).0
        }

        /// Wrapping (modular) negation. Computes `-self`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// negates `MIN` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is a positive
        /// value that is too large to represent in the type. In such
        /// a case, this function returns `MIN` itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_neg(), -100);
        /// assert_eq!((-128i8).wrapping_neg(), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_neg(self) -> Self {
            self.overflowing_neg().0
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(1u8.wrapping_shl(7), 128);
        /// assert_eq!(1u8.wrapping_shl(8), 1);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shl(self, rhs: u32) -> Self {
            self.overflowing_shl(rhs).0
        }

        /// Panic-free bitwise shift-left; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(128u8.wrapping_shr(7), 1);
        /// assert_eq!(128u8.wrapping_shr(8), 128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shr(self, rhs: u32) -> Self {
            self.overflowing_shr(rhs).0
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let x: i32 = 2; // or any other integer type
        ///
        /// assert_eq!(x.pow(4), 16);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn pow(self, mut exp: u32) -> Self {
            let mut base = self;
            let mut acc = Self::one();

            while exp > 1 {
                if (exp & 1) == 1 {
                    acc = acc * base;
                }
                exp /= 2;
                base = base * base;
            }

            // Deal with the final bit of the exponent separately, since
            // squaring the base afterwards is not necessary and may cause a
            // needless overflow.
            if exp == 1 {
                acc = acc * base;
            }

            acc
        }

        /// Computes the absolute value of `self`.
        ///
        /// # Overflow behavior
        ///
        /// The absolute value of `i32::min_value()` cannot be represented as an
        /// `i32`, and attempting to calculate it will cause an overflow. This
        /// means that code in debug mode will trigger a panic on this case and
        /// optimized code will return `i32::min_value()` without a panic.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(10i8.abs(), 10);
        /// assert_eq!((-10i8).abs(), 10);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn abs(self) -> Self {
            if self.is_negative() {
                // Note that the #[inline] above means that the overflow
                // semantics of this negation depend on the crate we're being
                // inlined into.
                -self
            } else {
                self
            }
        }

        /// Returns a number representing sign of `self`.
        ///
        /// - `0` if the number is zero
        /// - `1` if the number is positive
        /// - `-1` if the number is negative
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(10i8.signum(), 1);
        /// assert_eq!(0i8.signum(), 0);
        /// assert_eq!((-10i8).signum(), -1);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn signum(self) -> Self {
            match self {
                n if n > 0 =>  1,
                0          =>  0,
                _          => -1,
            }
        }

        /// Returns `true` if `self` is positive and `false` if the number
        /// is zero or negative.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert!(10i8.is_positive());
        /// assert!(!(-10i8).is_positive());
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_positive(self) -> bool { self > 0 }

        /// Returns `true` if `self` is negative and `false` if the number
        /// is zero or positive.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert!((-10i8).is_negative());
        /// assert!(!10i8.is_negative());
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_negative(self) -> bool { self < 0 }
    }
}

#[lang = "i8"]
#[cfg(stage0)]
impl i8 {
    int_impl! { i8, u8, 8,
        intrinsics::i8_add_with_overflow,
        intrinsics::i8_sub_with_overflow,
        intrinsics::i8_mul_with_overflow }
}
#[lang = "i8"]
#[cfg(not(stage0))]
impl i8 {
    int_impl! { i8, u8, 8,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i16"]
#[cfg(stage0)]
impl i16 {
    int_impl! { i16, u16, 16,
        intrinsics::i16_add_with_overflow,
        intrinsics::i16_sub_with_overflow,
        intrinsics::i16_mul_with_overflow }
}
#[lang = "i16"]
#[cfg(not(stage0))]
impl i16 {
    int_impl! { i16, u16, 16,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i32"]
#[cfg(stage0)]
impl i32 {
    int_impl! { i32, u32, 32,
        intrinsics::i32_add_with_overflow,
        intrinsics::i32_sub_with_overflow,
        intrinsics::i32_mul_with_overflow }
}
#[lang = "i32"]
#[cfg(not(stage0))]
impl i32 {
    int_impl! { i32, u32, 32,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i64"]
#[cfg(stage0)]
impl i64 {
    int_impl! { i64, u64, 64,
        intrinsics::i64_add_with_overflow,
        intrinsics::i64_sub_with_overflow,
        intrinsics::i64_mul_with_overflow }
}
#[lang = "i64"]
#[cfg(not(stage0))]
impl i64 {
    int_impl! { i64, u64, 64,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "32")]
#[lang = "isize"]
#[cfg(stage0)]
impl isize {
    int_impl! { i32, u32, 32,
        intrinsics::i32_add_with_overflow,
        intrinsics::i32_sub_with_overflow,
        intrinsics::i32_mul_with_overflow }
}
#[cfg(target_pointer_width = "32")]
#[lang = "isize"]
#[cfg(not(stage0))]
impl isize {
    int_impl! { i32, u32, 32,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "64")]
#[lang = "isize"]
#[cfg(stage0)]
impl isize {
    int_impl! { i64, u64, 64,
        intrinsics::i64_add_with_overflow,
        intrinsics::i64_sub_with_overflow,
        intrinsics::i64_mul_with_overflow }
}
#[cfg(target_pointer_width = "64")]
#[lang = "isize"]
#[cfg(not(stage0))]
impl isize {
    int_impl! { i64, u64, 64,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

// `Int` + `UnsignedInt` implemented for signed integers
macro_rules! uint_impl {
    ($ActualT:ty, $BITS:expr,
     $ctpop:path,
     $ctlz:path,
     $cttz:path,
     $bswap:path,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn min_value() -> Self { 0 }

        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn max_value() -> Self { !0 }

        /// Converts a string slice in a given base to an integer.
        ///
        /// Leading and trailing whitespace represent an error.
        ///
        /// # Arguments
        ///
        /// * src - A string slice
        /// * radix - The base to use. Must lie in the range [2 .. 36]
        ///
        /// # Return value
        ///
        /// `Err(ParseIntError)` if the string did not represent a valid number.
        /// Otherwise, `Ok(n)` where `n` is the integer represented by `src`.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
            from_str_radix(src, radix)
        }

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b01001100u8;
        ///
        /// assert_eq!(n.count_ones(), 3);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn count_ones(self) -> u32 {
            unsafe { $ctpop(self as $ActualT) as u32 }
        }

        /// Returns the number of zeros in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b01001100u8;
        ///
        /// assert_eq!(n.count_zeros(), 5);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn count_zeros(self) -> u32 {
            (!self).count_ones()
        }

        /// Returns the number of leading zeros in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b0101000u16;
        ///
        /// assert_eq!(n.leading_zeros(), 10);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn leading_zeros(self) -> u32 {
            unsafe { $ctlz(self as $ActualT) as u32 }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        #[cfg(stage0)]
        #[inline]
        pub fn trailing_zeros(self) -> u32 {
            // As of LLVM 3.6 the codegen for the zero-safe cttz8 intrinsic
            // emits two conditional moves on x86_64. By promoting the value to
            // u16 and setting bit 8, we get better code without any conditional
            // operations.
            // FIXME: There's a LLVM patch (http://reviews.llvm.org/D9284)
            // pending, remove this workaround once LLVM generates better code
            // for cttz8.
            unsafe {
                if $BITS == 8 {
                    intrinsics::cttz16(self as u16 | 0x100) as u32
                } else {
                    $cttz(self as $ActualT) as u32
                }
            }
        }
        /// Returns the number of trailing zeros in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0b0101000u16;
        ///
        /// assert_eq!(n.trailing_zeros(), 3);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[cfg(not(stage0))]
        #[inline]
        pub fn trailing_zeros(self) -> u32 {
            // As of LLVM 3.6 the codegen for the zero-safe cttz8 intrinsic
            // emits two conditional moves on x86_64. By promoting the value to
            // u16 and setting bit 8, we get better code without any conditional
            // operations.
            // FIXME: There's a LLVM patch (http://reviews.llvm.org/D9284)
            // pending, remove this workaround once LLVM generates better code
            // for cttz8.
            unsafe {
                if $BITS == 8 {
                    intrinsics::cttz(self as u16 | 0x100) as u32
                } else {
                    intrinsics::cttz(self) as u32
                }
            }
        }

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0x3456789ABCDEF012u64;
        ///
        /// assert_eq!(n.rotate_left(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_left(self, n: u32) -> Self {
            // Protect against undefined behaviour for over-long bit shifts
            let n = n % $BITS;
            (self << n) | (self >> (($BITS - n) % $BITS))
        }

        /// Shifts the bits to the right by a specified amount, `n`,
        /// wrapping the truncated bits to the beginning of the resulting
        /// integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xDEF0123456789ABCu64;
        ///
        /// assert_eq!(n.rotate_right(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_right(self, n: u32) -> Self {
            // Protect against undefined behaviour for over-long bit shifts
            let n = n % $BITS;
            (self >> n) | (self << (($BITS - n) % $BITS))
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xEFCDAB8967452301u64;
        ///
        /// assert_eq!(n.swap_bytes(), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn swap_bytes(self) -> Self {
            unsafe { $bswap(self as $ActualT) as Self }
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(u64::from_be(n), n)
        /// } else {
        ///     assert_eq!(u64::from_be(n), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn from_be(x: Self) -> Self {
            if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
        }

        /// Converts an integer from little endian to the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(u64::from_le(n), n)
        /// } else {
        ///     assert_eq!(u64::from_le(n), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn from_le(x: Self) -> Self {
            if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
        }

        /// Converts `self` to big endian from the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(n.to_be(), n)
        /// } else {
        ///     assert_eq!(n.to_be(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn to_be(self) -> Self { // or not to be?
            if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
        }

        /// Converts `self` to little endian from the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFu64;
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(n.to_le(), n)
        /// } else {
        ///     assert_eq!(n.to_le(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn to_le(self) -> Self {
            if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
        }

        /// Checked integer addition. Computes `self + other`, returning `None`
        /// if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(5u16.checked_add(65530), Some(65535));
        /// assert_eq!(6u16.checked_add(65530), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_add(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $add_with_overflow, self, other)
        }

        /// Checked integer subtraction. Computes `self - other`, returning
        /// `None` if underflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-127i8).checked_sub(1), Some(-128));
        /// assert_eq!((-128i8).checked_sub(1), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_sub(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $sub_with_overflow, self, other)
        }

        /// Checked integer multiplication. Computes `self * other`, returning
        /// `None` if underflow or overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(5u8.checked_mul(51), Some(255));
        /// assert_eq!(5u8.checked_mul(52), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_mul(self, other: Self) -> Option<Self> {
            checked_op!($ActualT, $mul_with_overflow, self, other)
        }

        /// Checked integer division. Computes `self / other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-127i8).checked_div(-1), Some(127));
        /// assert_eq!((-128i8).checked_div(-1), None);
        /// assert_eq!((1i8).checked_div(0), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_div(self, other: Self) -> Option<Self> {
            match other {
                0 => None,
                other => Some(self / other),
            }
        }

        /// Saturating integer addition. Computes `self + other`, saturating at
        /// the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.saturating_add(1), 101);
        /// assert_eq!(100i8.saturating_add(127), 127);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_add(self, other: Self) -> Self {
            match self.checked_add(other) {
                Some(x)                       => x,
                None if other >= Self::zero() => Self::max_value(),
                None => Self::min_value(),
            }
        }

        /// Saturating integer subtraction. Computes `self - other`, saturating
        /// at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.saturating_sub(127), -27);
        /// assert_eq!((-100i8).saturating_sub(127), -128);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_sub(self, other: Self) -> Self {
            match self.checked_sub(other) {
                Some(x)                       => x,
                None if other >= Self::zero() => Self::min_value(),
                None => Self::max_value(),
            }
        }

        /// Wrapping (modular) addition. Computes `self + other`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_add(27), 127);
        /// assert_eq!(100i8.wrapping_add(127), -29);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_add(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_add(self, rhs)
            }
        }

        /// Wrapping (modular) subtraction. Computes `self - other`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0i8.wrapping_sub(127), -127);
        /// assert_eq!((-2i8).wrapping_sub(127), 127);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_sub(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_sub(self, rhs)
            }
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// other`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(10i8.wrapping_mul(12), 120);
        /// assert_eq!(11i8.wrapping_mul(12), -124);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_mul(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_mul(self, rhs)
            }
        }

        /// Wrapping (modular) division. Computes `self / other`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// divides `MIN / -1` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is equivalent
        /// to `-MIN`, a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN`
        /// itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.wrapping_div(10), 10);
        /// assert_eq!((-128i8).wrapping_div(-1), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_div(self, rhs: Self) -> Self {
            self.overflowing_div(rhs).0
        }

        /// Wrapping (modular) remainder. Computes `self % other`,
        /// wrapping around at the boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically;
        /// implementation artifacts make `x % y` invalid for `MIN /
        /// -1` on a signed type (where `MIN` is the negative
        /// minimal value). In such a case, this function returns `0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_rem(10), 0);
        /// assert_eq!((-128i8).wrapping_rem(-1), 0);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_rem(self, rhs: Self) -> Self {
            self.overflowing_rem(rhs).0
        }

        /// Wrapping (modular) negation. Computes `-self`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// negates `MIN` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is a positive
        /// value that is too large to represent in the type. In such
        /// a case, this function returns `MIN` itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_neg(), -100);
        /// assert_eq!((-128i8).wrapping_neg(), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_neg(self) -> Self {
            self.overflowing_neg().0
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(1u8.wrapping_shl(7), 128);
        /// assert_eq!(1u8.wrapping_shl(8), 1);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shl(self, rhs: u32) -> Self {
            self.overflowing_shl(rhs).0
        }

        /// Panic-free bitwise shift-left; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(128u8.wrapping_shr(7), 1);
        /// assert_eq!(128u8.wrapping_shr(8), 128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shr(self, rhs: u32) -> Self {
            self.overflowing_shr(rhs).0
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(2i32.pow(4), 16);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn pow(self, mut exp: u32) -> Self {
            let mut base = self;
            let mut acc = Self::one();

            let mut prev_base = self;
            let mut base_oflo = false;
            while exp > 0 {
                if (exp & 1) == 1 {
                    if base_oflo {
                        // ensure overflow occurs in the same manner it
                        // would have otherwise (i.e. signal any exception
                        // it would have otherwise).
                        acc = acc * (prev_base * prev_base);
                    } else {
                        acc = acc * base;
                    }
                }
                prev_base = base;
                let (new_base, new_base_oflo) = base.overflowing_mul(base);
                base = new_base;
                base_oflo = new_base_oflo;
                exp /= 2;
            }
            acc
        }

        /// Returns `true` if and only if `self == 2^k` for some `k`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert!(16u8.is_power_of_two());
        /// assert!(!10u8.is_power_of_two());
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_power_of_two(self) -> bool {
            (self.wrapping_sub(Self::one())) & self == Self::zero() &&
                !(self == Self::zero())
        }

        /// Returns the smallest power of two greater than or equal to `self`.
        /// Unspecified behavior on overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(2u8.next_power_of_two(), 2);
        /// assert_eq!(3u8.next_power_of_two(), 4);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn next_power_of_two(self) -> Self {
            let bits = size_of::<Self>() * 8;
            let one: Self = Self::one();
            one << ((bits - self.wrapping_sub(one).leading_zeros() as usize) % bits)
        }

        /// Returns the smallest power of two greater than or equal to `n`. If
        /// the next power of two is greater than the type's maximum value,
        /// `None` is returned, otherwise the power of two is wrapped in `Some`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(2u8.checked_next_power_of_two(), Some(2));
        /// assert_eq!(3u8.checked_next_power_of_two(), Some(4));
        /// assert_eq!(200u8.checked_next_power_of_two(), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn checked_next_power_of_two(self) -> Option<Self> {
            let npot = self.next_power_of_two();
            if npot >= self {
                Some(npot)
            } else {
                None
            }
        }
    }
}

#[lang = "u8"]
#[cfg(stage0)]
impl u8 {
    uint_impl! { u8, 8,
        intrinsics::ctpop8,
        intrinsics::ctlz8,
        intrinsics::cttz8,
        bswap8,
        intrinsics::u8_add_with_overflow,
        intrinsics::u8_sub_with_overflow,
        intrinsics::u8_mul_with_overflow }
}
#[lang = "u8"]
#[cfg(not(stage0))]
impl u8 {
    uint_impl! { u8, 8,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "u16"]
#[cfg(stage0)]
impl u16 {
    uint_impl! { u16, 16,
        intrinsics::ctpop16,
        intrinsics::ctlz16,
        intrinsics::cttz16,
        intrinsics::bswap16,
        intrinsics::u16_add_with_overflow,
        intrinsics::u16_sub_with_overflow,
        intrinsics::u16_mul_with_overflow }
}
#[lang = "u16"]
#[cfg(not(stage0))]
impl u16 {
    uint_impl! { u16, 16,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "u32"]
#[cfg(stage0)]
impl u32 {
    uint_impl! { u32, 32,
        intrinsics::ctpop32,
        intrinsics::ctlz32,
        intrinsics::cttz32,
        intrinsics::bswap32,
        intrinsics::u32_add_with_overflow,
        intrinsics::u32_sub_with_overflow,
        intrinsics::u32_mul_with_overflow }
}
#[lang = "u32"]
#[cfg(not(stage0))]
impl u32 {
    uint_impl! { u32, 32,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "u64"]
#[cfg(stage0)]
impl u64 {
    uint_impl! { u64, 64,
        intrinsics::ctpop64,
        intrinsics::ctlz64,
        intrinsics::cttz64,
        intrinsics::bswap64,
        intrinsics::u64_add_with_overflow,
        intrinsics::u64_sub_with_overflow,
        intrinsics::u64_mul_with_overflow }
}
#[lang = "u64"]
#[cfg(not(stage0))]
impl u64 {
    uint_impl! { u64, 64,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "32")]
#[lang = "usize"]
#[cfg(stage0)]
impl usize {
    uint_impl! { u32, 32,
        intrinsics::ctpop32,
        intrinsics::ctlz32,
        intrinsics::cttz32,
        intrinsics::bswap32,
        intrinsics::u32_add_with_overflow,
        intrinsics::u32_sub_with_overflow,
        intrinsics::u32_mul_with_overflow }
}
#[cfg(target_pointer_width = "32")]
#[lang = "usize"]
#[cfg(not(stage0))]
impl usize {
    uint_impl! { u32, 32,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "64")]
#[lang = "usize"]
#[cfg(stage0)]
impl usize {
    uint_impl! { u64, 64,
        intrinsics::ctpop64,
        intrinsics::ctlz64,
        intrinsics::cttz64,
        intrinsics::bswap64,
        intrinsics::u64_add_with_overflow,
        intrinsics::u64_sub_with_overflow,
        intrinsics::u64_mul_with_overflow }
}
#[cfg(target_pointer_width = "64")]
#[lang = "usize"]
#[cfg(not(stage0))]
impl usize {
    uint_impl! { u64, 64,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

/// Used for representing the classification of floating point numbers
#[derive(Copy, Clone, PartialEq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum FpCategory {
    /// "Not a Number", often obtained by dividing by zero
    #[stable(feature = "rust1", since = "1.0.0")]
    Nan,

    /// Positive or negative infinity
    #[stable(feature = "rust1", since = "1.0.0")]
    Infinite ,

    /// Positive or negative zero
    #[stable(feature = "rust1", since = "1.0.0")]
    Zero,

    /// De-normalized floating point representation (less precise than `Normal`)
    #[stable(feature = "rust1", since = "1.0.0")]
    Subnormal,

    /// A regular floating point number
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal,
}

/// A built-in floating point number.
#[doc(hidden)]
#[unstable(feature = "core_float",
           reason = "stable interface is via `impl f{32,64}` in later crates",
           issue = "27702")]
pub trait Float: Sized {
    /// Returns the NaN value.
    fn nan() -> Self;
    /// Returns the infinite value.
    fn infinity() -> Self;
    /// Returns the negative infinite value.
    fn neg_infinity() -> Self;
    /// Returns -0.0.
    fn neg_zero() -> Self;
    /// Returns 0.0.
    fn zero() -> Self;
    /// Returns 1.0.
    fn one() -> Self;
    /// Parses the string `s` with the radix `r` as a float.
    fn from_str_radix(s: &str, r: u32) -> Result<Self, ParseFloatError>;

    /// Returns true if this value is NaN and false otherwise.
    fn is_nan(self) -> bool;
    /// Returns true if this value is positive infinity or negative infinity and
    /// false otherwise.
    fn is_infinite(self) -> bool;
    /// Returns true if this number is neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// Returns true if this number is neither zero, infinite, denormal, or NaN.
    fn is_normal(self) -> bool;
    /// Returns the category that this number falls into.
    fn classify(self) -> FpCategory;

    /// Returns the mantissa, exponent and sign as integers, respectively.
    fn integer_decode(self) -> (u64, i16, i8);

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    fn abs(self) -> Self;
    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    fn signum(self) -> Self;
    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    fn is_positive(self) -> bool;
    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    fn is_negative(self) -> bool;

    /// Take the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self;

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    fn powi(self, n: i32) -> Self;

    /// Convert radians to degrees.
    fn to_degrees(self) -> Self;
    /// Convert degrees to radians.
    fn to_radians(self) -> Self;
}

macro_rules! from_str_radix_int_impl {
    ($($t:ty)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl FromStr for $t {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, ParseIntError> {
                from_str_radix(src, 10)
            }
        }
    )*}
}
from_str_radix_int_impl! { isize i8 i16 i32 i64 usize u8 u16 u32 u64 }

#[doc(hidden)]
trait FromStrRadixHelper: PartialOrd + Copy {
    fn min_value() -> Self;
    fn from_u32(u: u32) -> Self;
    fn checked_mul(&self, other: u32) -> Option<Self>;
    fn checked_sub(&self, other: u32) -> Option<Self>;
    fn checked_add(&self, other: u32) -> Option<Self>;
}

macro_rules! doit {
    ($($t:ty)*) => ($(impl FromStrRadixHelper for $t {
        fn min_value() -> Self { Self::min_value() }
        fn from_u32(u: u32) -> Self { u as Self }
        fn checked_mul(&self, other: u32) -> Option<Self> {
            Self::checked_mul(*self, other as Self)
        }
        fn checked_sub(&self, other: u32) -> Option<Self> {
            Self::checked_sub(*self, other as Self)
        }
        fn checked_add(&self, other: u32) -> Option<Self> {
            Self::checked_add(*self, other as Self)
        }
    })*)
}
doit! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }

fn from_str_radix<T: FromStrRadixHelper>(src: &str, radix: u32)
                                         -> Result<T, ParseIntError> {
    use self::IntErrorKind::*;
    use self::ParseIntError as PIE;

    assert!(radix >= 2 && radix <= 36,
           "from_str_radix_int: must lie in the range `[2, 36]` - found {}",
           radix);

    if src.is_empty() {
        return Err(PIE { kind: Empty });
    }

    let is_signed_ty = T::from_u32(0) > T::min_value();

    // all valid digits are ascii, so we will just iterate over the utf8 bytes
    // and cast them to chars. .to_digit() will safely return None for anything
    // other than a valid ascii digit for the given radix, including the first-byte
    // of multi-byte sequences
    let src = src.as_bytes();

    let (is_positive, digits) = match src[0] {
        b'+' => (true, &src[1..]),
        b'-' if is_signed_ty => (false, &src[1..]),
        _ => (true, src)
    };

    if digits.is_empty() {
        return Err(PIE { kind: Empty });
    }

    let mut result = T::from_u32(0);
    if is_positive {
        // The number is positive
        for &c in digits {
            let x = match (c as char).to_digit(radix) {
                Some(x) => x,
                None => return Err(PIE { kind: InvalidDigit }),
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return Err(PIE { kind: Overflow }),
            };
            result = match result.checked_add(x) {
                Some(result) => result,
                None => return Err(PIE { kind: Overflow }),
            };
        }
    } else {
        // The number is negative
        for &c in digits {
            let x = match (c as char).to_digit(radix) {
                Some(x) => x,
                None => return Err(PIE { kind: InvalidDigit }),
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return Err(PIE { kind: Underflow }),
            };
            result = match result.checked_sub(x) {
                Some(result) => result,
                None => return Err(PIE { kind: Underflow }),
            };
        }
    }
    Ok(result)
}

/// An error which can be returned when parsing an integer.
#[derive(Debug, Clone, PartialEq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseIntError { kind: IntErrorKind }

#[derive(Debug, Clone, PartialEq)]
enum IntErrorKind {
    Empty,
    InvalidDigit,
    Overflow,
    Underflow,
}

impl ParseIntError {
    #[unstable(feature = "int_error_internals",
               reason = "available through Error trait and this method should \
                         not be exposed publicly",
               issue = "0")]
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        match self.kind {
            IntErrorKind::Empty => "cannot parse integer from empty string",
            IntErrorKind::InvalidDigit => "invalid digit found in string",
            IntErrorKind::Overflow => "number too large to fit in target type",
            IntErrorKind::Underflow => "number too small to fit in target type",
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseIntError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.__description().fmt(f)
    }
}

pub use num::dec2flt::ParseFloatError;

// Conversion traits for primitive integer and float types
// Conversions T -> T are covered by a blanket impl and therefore excluded
// Some conversions from and to usize/isize are not implemented due to portability concerns
macro_rules! impl_from {
    ($Small: ty, $Large: ty) => {
        #[stable(feature = "lossless_prim_conv", since = "1.5.0")]
        impl From<$Small> for $Large {
            #[inline]
            fn from(small: $Small) -> $Large {
                small as $Large
            }
        }
    }
}

// Unsigned -> Unsigned
impl_from! { u8, u16 }
impl_from! { u8, u32 }
impl_from! { u8, u64 }
impl_from! { u8, usize }
impl_from! { u16, u32 }
impl_from! { u16, u64 }
impl_from! { u32, u64 }

// Signed -> Signed
impl_from! { i8, i16 }
impl_from! { i8, i32 }
impl_from! { i8, i64 }
impl_from! { i8, isize }
impl_from! { i16, i32 }
impl_from! { i16, i64 }
impl_from! { i32, i64 }

// Unsigned -> Signed
impl_from! { u8, i16 }
impl_from! { u8, i32 }
impl_from! { u8, i64 }
impl_from! { u16, i32 }
impl_from! { u16, i64 }
impl_from! { u32, i64 }

// Note: integers can only be represented with full precision in a float if
// they fit in the significand, which is 24 bits in f32 and 53 bits in f64.
// Lossy float conversions are not implemented at this time.

// Signed -> Float
impl_from! { i8, f32 }
impl_from! { i8, f64 }
impl_from! { i16, f32 }
impl_from! { i16, f64 }
impl_from! { i32, f64 }

// Unsigned -> Float
impl_from! { u8, f32 }
impl_from! { u8, f64 }
impl_from! { u16, f32 }
impl_from! { u16, f64 }
impl_from! { u32, f64 }

// Float -> Float
impl_from! { f32, f64 }
