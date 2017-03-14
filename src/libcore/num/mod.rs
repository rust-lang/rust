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

use convert::TryFrom;
use fmt;
use intrinsics;
use mem::size_of;
use str::FromStr;

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
///
/// # Examples
///
/// ```
/// use std::num::Wrapping;
///
/// let zero = Wrapping(0u32);
/// let one = Wrapping(1u32);
///
/// assert_eq!(std::u32::MAX, (zero - one).0);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
pub struct Wrapping<T>(#[stable(feature = "rust1", since = "1.0.0")]
                       pub T);

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_display", since = "1.10.0")]
impl<T: fmt::Display> fmt::Display for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::Binary> fmt::Binary for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::Octal> fmt::Octal for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::LowerHex> fmt::LowerHex for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::UpperHex> fmt::UpperHex for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

mod wrapping;

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
#[rustc_deprecated(since = "1.11.0", reason = "no longer used for \
                                               Iterator::sum")]
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
#[rustc_deprecated(since = "1.11.0", reason = "no longer used for \
                                               Iterator::product")]
pub trait One: Sized {
    /// The "one" (usually, multiplicative identity) for this type.
    fn one() -> Self;
}

macro_rules! zero_one_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "zero_one",
                   reason = "unsure of placement, wants to use associated constants",
                   issue = "27739")]
        #[allow(deprecated)]
        impl Zero for $t {
            #[inline]
            fn zero() -> Self { 0 }
        }
        #[unstable(feature = "zero_one",
                   reason = "unsure of placement, wants to use associated constants",
                   issue = "27739")]
        #[allow(deprecated)]
        impl One for $t {
            #[inline]
            fn one() -> Self { 1 }
        }
    )*)
}
zero_one_impl! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

macro_rules! zero_one_impl_float {
    ($($t:ty)*) => ($(
        #[unstable(feature = "zero_one",
                   reason = "unsure of placement, wants to use associated constants",
                   issue = "27739")]
        #[allow(deprecated)]
        impl Zero for $t {
            #[inline]
            fn zero() -> Self { 0.0 }
        }
        #[unstable(feature = "zero_one",
                   reason = "unsure of placement, wants to use associated constants",
                   issue = "27739")]
        #[allow(deprecated)]
        impl One for $t {
            #[inline]
            fn one() -> Self { 1.0 }
        }
    )*)
}
zero_one_impl_float! { f32 f64 }

macro_rules! checked_op {
    ($U:ty, $op:path, $x:expr, $y:expr) => {{
        let (result, overflowed) = unsafe { $op($x as $U, $y as $U) };
        if overflowed { None } else { Some(result as Self) }
    }}
}

// `Int` + `SignedInt` implemented for signed integers
macro_rules! int_impl {
    ($ActualT:ident, $UnsignedT:ty, $BITS:expr,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        /// Returns the smallest value that can be represented by this integer type.
        ///
        /// # Examples
        ///
        /// ```
        /// assert_eq!(i8::min_value(), -128);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub const fn min_value() -> Self {
            !0 ^ ((!0 as $UnsignedT) >> 1) as Self
        }

        /// Returns the largest value that can be represented by this integer type.
        ///
        /// # Examples
        ///
        /// ```
        /// assert_eq!(i8::max_value(), 127);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub const fn max_value() -> Self {
            !Self::min_value()
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
        /// assert_eq!(i32::from_str_radix("A", 16), Ok(10));
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
        /// let n = -0b1000_0000i8;
        ///
        /// assert_eq!(n.count_ones(), 1);
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
        /// let n = -0b1000_0000i8;
        ///
        /// assert_eq!(n.count_zeros(), 7);
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
        /// let n = -1i16;
        ///
        /// assert_eq!(n.leading_zeros(), 0);
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
        /// let n = -4i8;
        ///
        /// assert_eq!(n.trailing_zeros(), 2);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn trailing_zeros(self) -> u32 {
            (self as $UnsignedT).trailing_zeros()
        }

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// Please note this isn't the same operation as `<<`!
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFi64;
        /// let m = -0x76543210FEDCBA99i64;
        ///
        /// assert_eq!(n.rotate_left(32), m);
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
        /// Please note this isn't the same operation as `>>`!
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// let n = 0x0123456789ABCDEFi64;
        /// let m = -0xFEDCBA987654322i64;
        ///
        /// assert_eq!(n.rotate_right(4), m);
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
        /// let n =  0x0123456789ABCDEFi64;
        /// let m = -0x1032547698BADCFFi64;
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
        /// let n = 0x0123456789ABCDEFi64;
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(i64::from_be(n), n)
        /// } else {
        ///     assert_eq!(i64::from_be(n), n.swap_bytes())
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
        /// let n = 0x0123456789ABCDEFi64;
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(i64::from_le(n), n)
        /// } else {
        ///     assert_eq!(i64::from_le(n), n.swap_bytes())
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
        /// let n = 0x0123456789ABCDEFi64;
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
        /// let n = 0x0123456789ABCDEFi64;
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
        /// assert_eq!(7i16.checked_add(32760), Some(32767));
        /// assert_eq!(8i16.checked_add(32760), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_add(self, other: Self) -> Option<Self> {
            let (a, b) = self.overflowing_add(other);
            if b {None} else {Some(a)}
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
            let (a, b) = self.overflowing_sub(other);
            if b {None} else {Some(a)}
        }

        /// Checked integer multiplication. Computes `self * other`, returning
        /// `None` if underflow or overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(6i8.checked_mul(21), Some(126));
        /// assert_eq!(6i8.checked_mul(22), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_mul(self, other: Self) -> Option<Self> {
            let (a, b) = self.overflowing_mul(other);
            if b {None} else {Some(a)}
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
            if other == 0 || (self == Self::min_value() && other == -1) {
                None
            } else {
                Some(unsafe { intrinsics::unchecked_div(self, other) })
            }
        }

        /// Checked integer remainder. Computes `self % other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.checked_rem(2), Some(1));
        /// assert_eq!(5i32.checked_rem(0), None);
        /// assert_eq!(i32::MIN.checked_rem(-1), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_rem(self, other: Self) -> Option<Self> {
            if other == 0 || (self == Self::min_value() && other == -1) {
                None
            } else {
                Some(unsafe { intrinsics::unchecked_rem(self, other) })
            }
        }

        /// Checked negation. Computes `-self`, returning `None` if `self ==
        /// MIN`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.checked_neg(), Some(-5));
        /// assert_eq!(i32::MIN.checked_neg(), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_neg(self) -> Option<Self> {
            let (a, b) = self.overflowing_neg();
            if b {None} else {Some(a)}
        }

        /// Checked shift left. Computes `self << rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0x10i32.checked_shl(4), Some(0x100));
        /// assert_eq!(0x10i32.checked_shl(33), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_shl(self, rhs: u32) -> Option<Self> {
            let (a, b) = self.overflowing_shl(rhs);
            if b {None} else {Some(a)}
        }

        /// Checked shift right. Computes `self >> rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0x10i32.checked_shr(4), Some(0x1));
        /// assert_eq!(0x10i32.checked_shr(33), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_shr(self, rhs: u32) -> Option<Self> {
            let (a, b) = self.overflowing_shr(rhs);
            if b {None} else {Some(a)}
        }

        /// Checked absolute value. Computes `self.abs()`, returning `None` if
        /// `self == MIN`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!((-5i32).checked_abs(), Some(5));
        /// assert_eq!(i32::MIN.checked_abs(), None);
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[inline]
        pub fn checked_abs(self) -> Option<Self> {
            if self.is_negative() {
                self.checked_neg()
            } else {
                Some(self)
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
                Some(x) => x,
                None if other >= 0 => Self::max_value(),
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
                Some(x) => x,
                None if other >= 0 => Self::min_value(),
                None => Self::max_value(),
            }
        }

        /// Saturating integer multiplication. Computes `self * other`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(100i32.saturating_mul(127), 12700);
        /// assert_eq!((1i32 << 23).saturating_mul(1 << 23), i32::MAX);
        /// assert_eq!((-1i32 << 23).saturating_mul(1 << 23), i32::MIN);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn saturating_mul(self, other: Self) -> Self {
            self.checked_mul(other).unwrap_or_else(|| {
                if (self < 0 && other < 0) || (self > 0 && other > 0) {
                    Self::max_value()
                } else {
                    Self::min_value()
                }
            })
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
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
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
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
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
        /// Note that this is *not* the same as a rotate-left; the
        /// RHS of a wrapping shift-left is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a `rotate_left` function, which may
        /// be what you want instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-1i8).wrapping_shl(7), -128);
        /// assert_eq!((-1i8).wrapping_shl(8), -1);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shl(self, rhs: u32) -> Self {
            self.overflowing_shl(rhs).0
        }

        /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-right; the
        /// RHS of a wrapping shift-right is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a `rotate_right` function, which may
        /// be what you want instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!((-128i8).wrapping_shr(7), -1);
        /// assert_eq!((-128i8).wrapping_shr(8), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_shr(self, rhs: u32) -> Self {
            self.overflowing_shr(rhs).0
        }

        /// Wrapping (modular) absolute value. Computes `self.abs()`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one takes
        /// the absolute value of the negative minimal value for the type
        /// this is a positive value that is too large to represent in the
        /// type. In such a case, this function returns `MIN` itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_abs(), 100);
        /// assert_eq!((-100i8).wrapping_abs(), 100);
        /// assert_eq!((-128i8).wrapping_abs(), -128);
        /// assert_eq!((-128i8).wrapping_abs() as u8, 128);
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[inline(always)]
        pub fn wrapping_abs(self) -> Self {
            if self.is_negative() {
                self.wrapping_neg()
            } else {
                self
            }
        }

        /// Calculates `self` + `rhs`
        ///
        /// Returns a tuple of the addition along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.overflowing_add(2), (7, false));
        /// assert_eq!(i32::MAX.overflowing_add(1), (i32::MIN, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $add_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates `self` - `rhs`
        ///
        /// Returns a tuple of the subtraction along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.overflowing_sub(2), (3, false));
        /// assert_eq!(i32::MIN.overflowing_sub(1), (i32::MAX, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $sub_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(5i32.overflowing_mul(2), (10, false));
        /// assert_eq!(1_000_000_000i32.overflowing_mul(10), (1410065408, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $mul_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates the divisor when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// occur then self is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.overflowing_div(2), (2, false));
        /// assert_eq!(i32::MIN.overflowing_div(-1), (i32::MIN, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_div(self, rhs: Self) -> (Self, bool) {
            if self == Self::min_value() && rhs == -1 {
                (self, true)
            } else {
                (self / rhs, false)
            }
        }

        /// Calculates the remainder when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would occur then 0 is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(5i32.overflowing_rem(2), (1, false));
        /// assert_eq!(i32::MIN.overflowing_rem(-1), (0, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
            if self == Self::min_value() && rhs == -1 {
                (0, true)
            } else {
                (self % rhs, false)
            }
        }

        /// Negates self, overflowing if this is equal to the minimum value.
        ///
        /// Returns a tuple of the negated version of self along with a boolean
        /// indicating whether an overflow happened. If `self` is the minimum
        /// value (e.g. `i32::MIN` for values of type `i32`), then the minimum
        /// value will be returned again and `true` will be returned for an
        /// overflow happening.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::i32;
        ///
        /// assert_eq!(2i32.overflowing_neg(), (-2, false));
        /// assert_eq!(i32::MIN.overflowing_neg(), (i32::MIN, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_neg(self) -> (Self, bool) {
            if self == Self::min_value() {
                (Self::min_value(), true)
            } else {
                (-self, false)
            }
        }

        /// Shifts self left by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(0x10i32.overflowing_shl(4), (0x100, false));
        /// assert_eq!(0x10i32.overflowing_shl(36), (0x100, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
            (self << (rhs & ($BITS - 1)), (rhs > ($BITS - 1)))
        }

        /// Shifts self right by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(0x10i32.overflowing_shr(4), (0x1, false));
        /// assert_eq!(0x10i32.overflowing_shr(36), (0x1, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
            (self >> (rhs & ($BITS - 1)), (rhs > ($BITS - 1)))
        }

        /// Computes the absolute value of `self`.
        ///
        /// Returns a tuple of the absolute version of self along with a
        /// boolean indicating whether an overflow happened. If self is the
        /// minimum value (e.g. i32::MIN for values of type i32), then the
        /// minimum value will be returned again and true will be returned for
        /// an overflow happening.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(10i8.overflowing_abs(), (10,false));
        /// assert_eq!((-10i8).overflowing_abs(), (10,false));
        /// assert_eq!((-128i8).overflowing_abs(), (-128,true));
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[inline]
        pub fn overflowing_abs(self) -> (Self, bool) {
            if self.is_negative() {
                self.overflowing_neg()
            } else {
                (self, false)
            }
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
        #[rustc_inherit_overflow_checks]
        pub fn pow(self, mut exp: u32) -> Self {
            let mut base = self;
            let mut acc = 1;

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
        #[rustc_inherit_overflow_checks]
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
impl i8 {
    int_impl! { i8, u8, 8,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i16"]
impl i16 {
    int_impl! { i16, u16, 16,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i32"]
impl i32 {
    int_impl! { i32, u32, 32,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i64"]
impl i64 {
    int_impl! { i64, u64, 64,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[lang = "i128"]
impl i128 {
    int_impl! { i128, u128, 128,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "16")]
#[lang = "isize"]
impl isize {
    int_impl! { i16, u16, 16,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "32")]
#[lang = "isize"]
impl isize {
    int_impl! { i32, u32, 32,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "64")]
#[lang = "isize"]
impl isize {
    int_impl! { i64, u64, 64,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

// `Int` + `UnsignedInt` implemented for unsigned integers
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
        ///
        /// # Examples
        ///
        /// ```
        /// assert_eq!(u8::min_value(), 0);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub const fn min_value() -> Self { 0 }

        /// Returns the largest value that can be represented by this integer type.
        ///
        /// # Examples
        ///
        /// ```
        /// assert_eq!(u8::max_value(), 255);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub const fn max_value() -> Self { !0 }

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
        /// Please note this isn't the same operation as `<<`!
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
        /// Please note this isn't the same operation as `>>`!
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
            let (a, b) = self.overflowing_add(other);
            if b {None} else {Some(a)}
        }

        /// Checked integer subtraction. Computes `self - other`, returning
        /// `None` if underflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(1u8.checked_sub(1), Some(0));
        /// assert_eq!(0u8.checked_sub(1), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_sub(self, other: Self) -> Option<Self> {
            let (a, b) = self.overflowing_sub(other);
            if b {None} else {Some(a)}
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
            let (a, b) = self.overflowing_mul(other);
            if b {None} else {Some(a)}
        }

        /// Checked integer division. Computes `self / other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(128u8.checked_div(2), Some(64));
        /// assert_eq!(1u8.checked_div(0), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_div(self, other: Self) -> Option<Self> {
            match other {
                0 => None,
                other => Some(unsafe { intrinsics::unchecked_div(self, other) }),
            }
        }

        /// Checked integer remainder. Computes `self % other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(5u32.checked_rem(2), Some(1));
        /// assert_eq!(5u32.checked_rem(0), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_rem(self, other: Self) -> Option<Self> {
            if other == 0 {
                None
            } else {
                Some(unsafe { intrinsics::unchecked_rem(self, other) })
            }
        }

        /// Checked negation. Computes `-self`, returning `None` unless `self ==
        /// 0`.
        ///
        /// Note that negating any positive integer will overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0u32.checked_neg(), Some(0));
        /// assert_eq!(1u32.checked_neg(), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_neg(self) -> Option<Self> {
            let (a, b) = self.overflowing_neg();
            if b {None} else {Some(a)}
        }

        /// Checked shift left. Computes `self << rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0x10u32.checked_shl(4), Some(0x100));
        /// assert_eq!(0x10u32.checked_shl(33), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_shl(self, rhs: u32) -> Option<Self> {
            let (a, b) = self.overflowing_shl(rhs);
            if b {None} else {Some(a)}
        }

        /// Checked shift right. Computes `self >> rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(0x10u32.checked_shr(4), Some(0x1));
        /// assert_eq!(0x10u32.checked_shr(33), None);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn checked_shr(self, rhs: u32) -> Option<Self> {
            let (a, b) = self.overflowing_shr(rhs);
            if b {None} else {Some(a)}
        }

        /// Saturating integer addition. Computes `self + other`, saturating at
        /// the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.saturating_add(1), 101);
        /// assert_eq!(200u8.saturating_add(127), 255);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_add(self, other: Self) -> Self {
            match self.checked_add(other) {
                Some(x) => x,
                None => Self::max_value(),
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
        /// assert_eq!(100u8.saturating_sub(27), 73);
        /// assert_eq!(13u8.saturating_sub(127), 0);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_sub(self, other: Self) -> Self {
            match self.checked_sub(other) {
                Some(x) => x,
                None => Self::min_value(),
            }
        }

        /// Saturating integer multiplication. Computes `self * other`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// use std::u32;
        ///
        /// assert_eq!(100u32.saturating_mul(127), 12700);
        /// assert_eq!((1u32 << 23).saturating_mul(1 << 23), u32::MAX);
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[inline]
        pub fn saturating_mul(self, other: Self) -> Self {
            self.checked_mul(other).unwrap_or(Self::max_value())
        }

        /// Wrapping (modular) addition. Computes `self + other`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(200u8.wrapping_add(55), 255);
        /// assert_eq!(200u8.wrapping_add(155), 99);
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
        /// assert_eq!(100u8.wrapping_sub(100), 0);
        /// assert_eq!(100u8.wrapping_sub(155), 201);
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
        /// assert_eq!(10u8.wrapping_mul(12), 120);
        /// assert_eq!(25u8.wrapping_mul(12), 44);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_mul(self, rhs: Self) -> Self {
            unsafe {
                intrinsics::overflowing_mul(self, rhs)
            }
        }

        /// Wrapping (modular) division. Computes `self / other`.
        /// Wrapped division on unsigned types is just normal division.
        /// There's no way wrapping could ever happen.
        /// This function exists, so that all operations
        /// are accounted for in the wrapping operations.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.wrapping_div(10), 10);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_div(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Wrapping (modular) remainder. Computes `self % other`.
        /// Wrapped remainder calculation on unsigned types is
        /// just the regular remainder calculation.
        /// There's no way wrapping could ever happen.
        /// This function exists, so that all operations
        /// are accounted for in the wrapping operations.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.wrapping_rem(10), 0);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[inline(always)]
        pub fn wrapping_rem(self, rhs: Self) -> Self {
            self % rhs
        }

        /// Wrapping (modular) negation. Computes `-self`,
        /// wrapping around at the boundary of the type.
        ///
        /// Since unsigned types do not have negative equivalents
        /// all applications of this function will wrap (except for `-0`).
        /// For values smaller than the corresponding signed type's maximum
        /// the result is the same as casting the corresponding signed value.
        /// Any larger values are equivalent to `MAX + 1 - (val - MAX - 1)` where
        /// `MAX` is the corresponding signed type's maximum.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(100u8.wrapping_neg(), 156);
        /// assert_eq!(0u8.wrapping_neg(), 0);
        /// assert_eq!(180u8.wrapping_neg(), 76);
        /// assert_eq!(180u8.wrapping_neg(), (127 + 1) - (180u8 - (127 + 1)));
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
        /// Note that this is *not* the same as a rotate-left; the
        /// RHS of a wrapping shift-left is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a `rotate_left` function, which may
        /// be what you want instead.
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

        /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-right; the
        /// RHS of a wrapping shift-right is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a `rotate_right` function, which may
        /// be what you want instead.
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

        /// Calculates `self` + `rhs`
        ///
        /// Returns a tuple of the addition along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::u32;
        ///
        /// assert_eq!(5u32.overflowing_add(2), (7, false));
        /// assert_eq!(u32::MAX.overflowing_add(1), (0, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $add_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates `self` - `rhs`
        ///
        /// Returns a tuple of the subtraction along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// use std::u32;
        ///
        /// assert_eq!(5u32.overflowing_sub(2), (3, false));
        /// assert_eq!(0u32.overflowing_sub(1), (u32::MAX, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $sub_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(5u32.overflowing_mul(2), (10, false));
        /// assert_eq!(1_000_000_000u32.overflowing_mul(10), (1410065408, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
            unsafe {
                let (a, b) = $mul_with_overflow(self as $ActualT,
                                                rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates the divisor when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. Note that for unsigned
        /// integers overflow never occurs, so the second value is always
        /// `false`.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(5u32.overflowing_div(2), (2, false));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_div(self, rhs: Self) -> (Self, bool) {
            (self / rhs, false)
        }

        /// Calculates the remainder when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. Note that for
        /// unsigned integers overflow never occurs, so the second value is
        /// always `false`.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is 0.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(5u32.overflowing_rem(2), (1, false));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
            (self % rhs, false)
        }

        /// Negates self in an overflowing fashion.
        ///
        /// Returns `!self + 1` using wrapping operations to return the value
        /// that represents the negation of this unsigned value. Note that for
        /// positive unsigned values overflow always occurs, but negating 0 does
        /// not overflow.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(0u32.overflowing_neg(), (0, false));
        /// assert_eq!(2u32.overflowing_neg(), (-2i32 as u32, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_neg(self) -> (Self, bool) {
            ((!self).wrapping_add(1), self != 0)
        }

        /// Shifts self left by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(0x10u32.overflowing_shl(4), (0x100, false));
        /// assert_eq!(0x10u32.overflowing_shl(36), (0x100, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
            (self << (rhs & ($BITS - 1)), (rhs > ($BITS - 1)))
        }

        /// Shifts self right by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage
        ///
        /// ```
        /// assert_eq!(0x10u32.overflowing_shr(4), (0x1, false));
        /// assert_eq!(0x10u32.overflowing_shr(36), (0x1, true));
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        pub fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
            (self >> (rhs & ($BITS - 1)), (rhs > ($BITS - 1)))
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// assert_eq!(2u32.pow(4), 16);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub fn pow(self, mut exp: u32) -> Self {
            let mut base = self;
            let mut acc = 1;

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
            (self.wrapping_sub(1)) & self == 0 && !(self == 0)
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
            let one: Self = 1;
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

#[lang = "u128"]
impl u128 {
    uint_impl! { u128, 128,
        intrinsics::ctpop,
        intrinsics::ctlz,
        intrinsics::cttz,
        intrinsics::bswap,
        intrinsics::add_with_overflow,
        intrinsics::sub_with_overflow,
        intrinsics::mul_with_overflow }
}

#[cfg(target_pointer_width = "16")]
#[lang = "usize"]
impl usize {
    uint_impl! { u16, 16,
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

/// A classification of floating point numbers.
///
/// This `enum` is used as the return type for [`f32::classify`] and [`f64::classify`]. See
/// their documentation for more.
///
/// [`f32::classify`]: ../../std/primitive.f32.html#method.classify
/// [`f64::classify`]: ../../std/primitive.f64.html#method.classify
///
/// # Examples
///
/// ```
/// use std::num::FpCategory;
/// use std::f32;
///
/// let num = 12.4_f32;
/// let inf = f32::INFINITY;
/// let zero = 0f32;
/// let sub: f32 = 1.1754942e-38;
/// let nan = f32::NAN;
///
/// assert_eq!(num.classify(), FpCategory::Normal);
/// assert_eq!(inf.classify(), FpCategory::Infinite);
/// assert_eq!(zero.classify(), FpCategory::Zero);
/// assert_eq!(nan.classify(), FpCategory::Nan);
/// assert_eq!(sub.classify(), FpCategory::Subnormal);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum FpCategory {
    /// "Not a Number", often obtained by dividing by zero.
    #[stable(feature = "rust1", since = "1.0.0")]
    Nan,

    /// Positive or negative infinity.
    #[stable(feature = "rust1", since = "1.0.0")]
    Infinite,

    /// Positive or negative zero.
    #[stable(feature = "rust1", since = "1.0.0")]
    Zero,

    /// De-normalized floating point representation (less precise than `Normal`).
    #[stable(feature = "rust1", since = "1.0.0")]
    Subnormal,

    /// A regular floating point number.
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal,
}

/// A built-in floating point number.
#[doc(hidden)]
#[unstable(feature = "core_float",
           reason = "stable interface is via `impl f{32,64}` in later crates",
           issue = "32110")]
pub trait Float: Sized {
    /// Returns the NaN value.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn nan() -> Self;
    /// Returns the infinite value.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn infinity() -> Self;
    /// Returns the negative infinite value.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn neg_infinity() -> Self;
    /// Returns -0.0.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn neg_zero() -> Self;
    /// Returns 0.0.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn zero() -> Self;
    /// Returns 1.0.
    #[unstable(feature = "float_extras", reason = "needs removal",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn one() -> Self;

    /// Returns true if this value is NaN and false otherwise.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_nan(self) -> bool;
    /// Returns true if this value is positive infinity or negative infinity and
    /// false otherwise.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_infinite(self) -> bool;
    /// Returns true if this number is neither infinite nor NaN.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_finite(self) -> bool;
    /// Returns true if this number is neither zero, infinite, denormal, or NaN.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_normal(self) -> bool;
    /// Returns the category that this number falls into.
    #[stable(feature = "core", since = "1.6.0")]
    fn classify(self) -> FpCategory;

    /// Returns the mantissa, exponent and sign as integers, respectively.
    #[unstable(feature = "float_extras", reason = "signature is undecided",
               issue = "27752")]
    #[rustc_deprecated(since = "1.11.0",
                       reason = "never really came to fruition and easily \
                                 implementable outside the standard library")]
    fn integer_decode(self) -> (u64, i16, i8);

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    #[stable(feature = "core", since = "1.6.0")]
    fn abs(self) -> Self;
    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    #[stable(feature = "core", since = "1.6.0")]
    fn signum(self) -> Self;

    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_sign_positive(self) -> bool;
    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    #[stable(feature = "core", since = "1.6.0")]
    fn is_sign_negative(self) -> bool;

    /// Take the reciprocal (inverse) of a number, `1/x`.
    #[stable(feature = "core", since = "1.6.0")]
    fn recip(self) -> Self;

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    #[stable(feature = "core", since = "1.6.0")]
    fn powi(self, n: i32) -> Self;

    /// Convert radians to degrees.
    #[stable(feature = "deg_rad_conversions", since="1.7.0")]
    fn to_degrees(self) -> Self;
    /// Convert degrees to radians.
    #[stable(feature = "deg_rad_conversions", since="1.7.0")]
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
from_str_radix_int_impl! { isize i8 i16 i32 i64 i128 usize u8 u16 u32 u64 u128 }

/// The error type returned when a checked integral type conversion fails.
#[unstable(feature = "try_from", issue = "33417")]
#[derive(Debug, Copy, Clone)]
pub struct TryFromIntError(());

impl TryFromIntError {
    #[unstable(feature = "int_error_internals",
               reason = "available through Error trait and this method should \
                         not be exposed publicly",
               issue = "0")]
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        "out of range integral type conversion attempted"
    }
}

#[unstable(feature = "try_from", issue = "33417")]
impl fmt::Display for TryFromIntError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.__description().fmt(fmt)
    }
}

macro_rules! same_sign_try_from_int_impl {
    ($storage:ty, $target:ty, $($source:ty),*) => {$(
        #[unstable(feature = "try_from", issue = "33417")]
        impl TryFrom<$source> for $target {
            type Err = TryFromIntError;

            fn try_from(u: $source) -> Result<$target, TryFromIntError> {
                let min = <$target as FromStrRadixHelper>::min_value() as $storage;
                let max = <$target as FromStrRadixHelper>::max_value() as $storage;
                if u as $storage < min || u as $storage > max {
                    Err(TryFromIntError(()))
                } else {
                    Ok(u as $target)
                }
            }
        }
    )*}
}

same_sign_try_from_int_impl!(u128, u8, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, i8, i8, i16, i32, i64, i128, isize);
same_sign_try_from_int_impl!(u128, u16, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, i16, i8, i16, i32, i64, i128, isize);
same_sign_try_from_int_impl!(u128, u32, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, i32, i8, i16, i32, i64, i128, isize);
same_sign_try_from_int_impl!(u128, u64, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, i64, i8, i16, i32, i64, i128, isize);
same_sign_try_from_int_impl!(u128, u128, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, i128, i8, i16, i32, i64, i128, isize);
same_sign_try_from_int_impl!(u128, usize, u8, u16, u32, u64, u128, usize);
same_sign_try_from_int_impl!(i128, isize, i8, i16, i32, i64, i128, isize);

macro_rules! cross_sign_from_int_impl {
    ($unsigned:ty, $($signed:ty),*) => {$(
        #[unstable(feature = "try_from", issue = "33417")]
        impl TryFrom<$unsigned> for $signed {
            type Err = TryFromIntError;

            fn try_from(u: $unsigned) -> Result<$signed, TryFromIntError> {
                let max = <$signed as FromStrRadixHelper>::max_value() as u128;
                if u as u128 > max {
                    Err(TryFromIntError(()))
                } else {
                    Ok(u as $signed)
                }
            }
        }

        #[unstable(feature = "try_from", issue = "33417")]
        impl TryFrom<$signed> for $unsigned {
            type Err = TryFromIntError;

            fn try_from(u: $signed) -> Result<$unsigned, TryFromIntError> {
                let max = <$unsigned as FromStrRadixHelper>::max_value() as u128;
                if u < 0 || u as u128 > max {
                    Err(TryFromIntError(()))
                } else {
                    Ok(u as $unsigned)
                }
            }
        }
    )*}
}

cross_sign_from_int_impl!(u8, i8, i16, i32, i64, i128, isize);
cross_sign_from_int_impl!(u16, i8, i16, i32, i64, i128, isize);
cross_sign_from_int_impl!(u32, i8, i16, i32, i64, i128, isize);
cross_sign_from_int_impl!(u64, i8, i16, i32, i64, i128, isize);
cross_sign_from_int_impl!(u128, i8, i16, i32, i64, i128, isize);
cross_sign_from_int_impl!(usize, i8, i16, i32, i64, i128, isize);

#[doc(hidden)]
trait FromStrRadixHelper: PartialOrd + Copy {
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn from_u32(u: u32) -> Self;
    fn checked_mul(&self, other: u32) -> Option<Self>;
    fn checked_sub(&self, other: u32) -> Option<Self>;
    fn checked_add(&self, other: u32) -> Option<Self>;
}

macro_rules! doit {
    ($($t:ty)*) => ($(impl FromStrRadixHelper for $t {
        fn min_value() -> Self { Self::min_value() }
        fn max_value() -> Self { Self::max_value() }
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
doit! { i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize }

fn from_str_radix<T: FromStrRadixHelper>(src: &str, radix: u32) -> Result<T, ParseIntError> {
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
        _ => (true, src),
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
///
/// This error is used as the error type for the `from_str_radix()` functions
/// on the primitive integer types, such as [`i8::from_str_radix`].
///
/// [`i8::from_str_radix`]: ../../std/primitive.i8.html#method.from_str_radix
#[derive(Debug, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseIntError {
    kind: IntErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[stable(feature = "rust1", since = "1.0.0")]
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
impl_from! { u8, u128 }
impl_from! { u8, usize }
impl_from! { u16, u32 }
impl_from! { u16, u64 }
impl_from! { u16, u128 }
impl_from! { u32, u64 }
impl_from! { u32, u128 }
impl_from! { u64, u128 }

// Signed -> Signed
impl_from! { i8, i16 }
impl_from! { i8, i32 }
impl_from! { i8, i64 }
impl_from! { i8, i128 }
impl_from! { i8, isize }
impl_from! { i16, i32 }
impl_from! { i16, i64 }
impl_from! { i16, i128 }
impl_from! { i32, i64 }
impl_from! { i32, i128 }
impl_from! { i64, i128 }

// Unsigned -> Signed
impl_from! { u8, i16 }
impl_from! { u8, i32 }
impl_from! { u8, i64 }
impl_from! { u8, i128 }
impl_from! { u16, i32 }
impl_from! { u16, i64 }
impl_from! { u16, i128 }
impl_from! { u32, i64 }
impl_from! { u32, i128 }
impl_from! { u64, i128 }

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
