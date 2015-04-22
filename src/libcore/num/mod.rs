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
use fmt;
use intrinsics;
use marker::Copy;
use mem::size_of;
use option::Option::{self, Some, None};
use result::Result::{self, Ok, Err};
use str::{FromStr, StrExt};

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
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Wrapping<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

#[unstable(feature = "core", reason = "may be removed or relocated")]
pub mod wrapping;

/// Types that have a "zero" value.
///
/// This trait is intended for use in conjunction with `Add`, as an identity:
/// `x + T::zero() == x`.
#[unstable(feature = "zero_one",
           reason = "unsure of placement, wants to use associated constants")]
pub trait Zero {
    /// The "zero" (usually, additive identity) for this type.
    fn zero() -> Self;
}

/// Types that have a "one" value.
///
/// This trait is intended for use in conjunction with `Mul`, as an identity:
/// `x * T::one() == x`.
#[unstable(feature = "zero_one",
           reason = "unsure of placement, wants to use associated constants")]
pub trait One {
    /// The "one" (usually, multiplicative identity) for this type.
    fn one() -> Self;
}

macro_rules! zero_one_impl {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            #[inline]
            fn zero() -> $t { 0 }
        }
        impl One for $t {
            #[inline]
            fn one() -> $t { 1 }
        }
    )*)
}
zero_one_impl! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

macro_rules! zero_one_impl_float {
    ($($t:ty)*) => ($(
        impl Zero for $t {
            #[inline]
            fn zero() -> $t { 0.0 }
        }
        impl One for $t {
            #[inline]
            fn one() -> $t { 1.0 }
        }
    )*)
}
zero_one_impl_float! { f32 f64 }

macro_rules! checked_op {
    ($T:ty, $U:ty, $op:path, $x:expr, $y:expr) => {{
        let (result, overflowed) = unsafe { $op($x as $U, $y as $U) };
        if overflowed { None } else { Some(result as $T) }
    }}
}

/// Swapping a single byte is a no-op. This is marked as `unsafe` for
/// consistency with the other `bswap` intrinsics.
unsafe fn bswap8(x: u8) -> u8 { x }

// `Int` + `SignedInt` implemented for signed integers
macro_rules! int_impl {
    ($T:ident = $ActualT:ty, $UnsignedT:ty, $BITS:expr,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn min_value() -> $T {
            (-1 as $T) << ($BITS - 1)
        }

        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn max_value() -> $T {
            let min = $T::min_value(); !min
        }

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
        #[allow(deprecated)]
        pub fn from_str_radix(src: &str, radix: u32) -> Result<$T, ParseIntError> {
            from_str_radix(src, radix)
        }

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// ```rust
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
        /// ```rust
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
        /// ```rust
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
        /// ```rust
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
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0x3456789ABCDEF012u64;
        ///
        /// assert_eq!(n.rotate_left(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_left(self, n: u32) -> $T {
            (self as $UnsignedT).rotate_left(n) as $T
        }

        /// Shifts the bits to the right by a specified amount, `n`,
        /// wrapping the truncated bits to the beginning of the resulting
        /// integer.
        ///
        /// # Examples
        ///
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xDEF0123456789ABCu64;
        ///
        /// assert_eq!(n.rotate_right(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_right(self, n: u32) -> $T {
            (self as $UnsignedT).rotate_right(n) as $T
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xEFCDAB8967452301u64;
        ///
        /// assert_eq!(n.swap_bytes(), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn swap_bytes(self) -> $T {
            (self as $UnsignedT).swap_bytes() as $T
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn from_be(x: $T) -> $T {
            if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
        }

        /// Converts an integer from little endian to the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn from_le(x: $T) -> $T {
            if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
        }

        /// Converts `self` to big endian from the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn to_be(self) -> $T { // or not to be?
            if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
        }

        /// Converts `self` to little endian from the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn to_le(self) -> $T {
            if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
        }

        /// Checked integer addition. Computes `self + other`, returning `None`
        /// if overflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!(5u16.checked_add(65530), Some(65535));
        /// assert_eq!(6u16.checked_add(65530), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_add(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $add_with_overflow, self, other)
        }

        /// Checked integer subtraction. Computes `self - other`, returning
        /// `None` if underflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!((-127i8).checked_sub(1), Some(-128));
        /// assert_eq!((-128i8).checked_sub(1), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_sub(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $sub_with_overflow, self, other)
        }

        /// Checked integer multiplication. Computes `self * other`, returning
        /// `None` if underflow or overflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!(5u8.checked_mul(51), Some(255));
        /// assert_eq!(5u8.checked_mul(52), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_mul(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $mul_with_overflow, self, other)
        }

        /// Checked integer division. Computes `self / other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!((-127i8).checked_div(-1), Some(127));
        /// assert_eq!((-128i8).checked_div(-1), None);
        /// assert_eq!((1i8).checked_div(0), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_div(self, v: $T) -> Option<$T> {
            match v {
                0   => None,
               -1 if self == <$T>::min_value()
                    => None,
                v   => Some(self / v),
            }
        }

        /// Saturating integer addition. Computes `self + other`, saturating at
        /// the numeric bounds instead of overflowing.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_add(self, other: $T) -> $T {
            match self.checked_add(other) {
                Some(x)                       => x,
                None if other >= <$T as Zero>::zero() => <$T>::max_value(),
                None => <$T>::min_value(),
            }
        }

        /// Saturating integer subtraction. Computes `self - other`, saturating
        /// at the numeric bounds instead of overflowing.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_sub(self, other: $T) -> $T {
            match self.checked_sub(other) {
                Some(x)                      => x,
                None if other >= <$T as Zero>::zero() => <$T>::min_value(),
                None => <$T>::max_value(),
            }
        }

        /// Wrapping (modular) addition. Computes `self + other`,
        /// wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_add(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_add(self, rhs)
            }
        }

        /// Wrapping (modular) subtraction. Computes `self - other`,
        /// wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_sub(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_sub(self, rhs)
            }
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// other`, wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_mul(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_mul(self, rhs)
            }
        }

        /// Wrapping (modular) division. Computes `floor(self / other)`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// divides `MIN / -1` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is equivalent
        /// to `-MIN`, a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN`
        /// itself..
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_div(self, rhs: $T) -> $T {
            self.overflowing_div(rhs).0
        }

        /// Wrapping (modular) remainder. Computes `self % other`,
        /// wrapping around at the boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically;
        /// implementation artifacts make `x % y` illegal for `MIN /
        /// -1` on a signed type illegal (where `MIN` is the negative
        /// minimal value). In such a case, this function returns `0`.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_rem(self, rhs: $T) -> $T {
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
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_neg(self) -> $T {
            self.overflowing_neg().0
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_shl(self, rhs: u32) -> $T {
            self.overflowing_shl(rhs).0
        }

        /// Panic-free bitwise shift-left; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_shr(self, rhs: u32) -> $T {
            self.overflowing_shr(rhs).0
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// ```
        /// let x: i32 = 2; // or any other integer type
        ///
        /// assert_eq!(x.pow(4), 16);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn pow(self, mut exp: u32) -> $T {
            let mut base = self;
            let mut acc = <$T as One>::one();

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

        /// Computes the absolute value of `self`. `Int::min_value()` will be
        /// returned if the number is `Int::min_value()`.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn abs(self) -> $T {
            if self.is_negative() {
                self.wrapping_neg()
            } else {
                self
            }
        }

        /// Returns a number representing sign of `self`.
        ///
        /// - `0` if the number is zero
        /// - `1` if the number is positive
        /// - `-1` if the number is negative
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn signum(self) -> $T {
            match self {
                n if n > 0 =>  1,
                0          =>  0,
                _          => -1,
            }
        }

        /// Returns `true` if `self` is positive and `false` if the number
        /// is zero or negative.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_positive(self) -> bool { self > 0 }

        /// Returns `true` if `self` is negative and `false` if the number
        /// is zero or positive.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_negative(self) -> bool { self < 0 }
    }
}

#[lang = "i8"]
impl i8 {
    int_impl! { i8 = i8, u8, 8,
        intrinsics::i8_add_with_overflow,
        intrinsics::i8_sub_with_overflow,
        intrinsics::i8_mul_with_overflow }
}

#[lang = "i16"]
impl i16 {
    int_impl! { i16 = i16, u16, 16,
        intrinsics::i16_add_with_overflow,
        intrinsics::i16_sub_with_overflow,
        intrinsics::i16_mul_with_overflow }
}

#[lang = "i32"]
impl i32 {
    int_impl! { i32 = i32, u32, 32,
        intrinsics::i32_add_with_overflow,
        intrinsics::i32_sub_with_overflow,
        intrinsics::i32_mul_with_overflow }
}

#[lang = "i64"]
impl i64 {
    int_impl! { i64 = i64, u64, 64,
        intrinsics::i64_add_with_overflow,
        intrinsics::i64_sub_with_overflow,
        intrinsics::i64_mul_with_overflow }
}

#[cfg(target_pointer_width = "32")]
#[lang = "isize"]
impl isize {
    int_impl! { isize = i32, u32, 32,
        intrinsics::i32_add_with_overflow,
        intrinsics::i32_sub_with_overflow,
        intrinsics::i32_mul_with_overflow }
}

#[cfg(target_pointer_width = "64")]
#[lang = "isize"]
impl isize {
    int_impl! { isize = i64, u64, 64,
        intrinsics::i64_add_with_overflow,
        intrinsics::i64_sub_with_overflow,
        intrinsics::i64_mul_with_overflow }
}

// `Int` + `UnsignedInt` implemented for signed integers
macro_rules! uint_impl {
    ($T:ty = $ActualT:ty, $BITS:expr,
     $ctpop:path,
     $ctlz:path,
     $cttz:path,
     $bswap:path,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn min_value() -> $T { 0 }

        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn max_value() -> $T { !0 }

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
        #[allow(deprecated)]
        pub fn from_str_radix(src: &str, radix: u32) -> Result<$T, ParseIntError> {
            from_str_radix(src, radix)
        }

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// ```rust
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
        /// ```rust
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
        /// ```rust
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
        /// ```rust
        /// let n = 0b0101000u16;
        ///
        /// assert_eq!(n.trailing_zeros(), 3);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn trailing_zeros(self) -> u32 {
            unsafe { $cttz(self as $ActualT) as u32 }
        }

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// # Examples
        ///
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0x3456789ABCDEF012u64;
        ///
        /// assert_eq!(n.rotate_left(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_left(self, n: u32) -> $T {
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
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xDEF0123456789ABCu64;
        ///
        /// assert_eq!(n.rotate_right(12), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn rotate_right(self, n: u32) -> $T {
            // Protect against undefined behaviour for over-long bit shifts
            let n = n % $BITS;
            (self >> n) | (self << (($BITS - n) % $BITS))
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// ```rust
        /// let n = 0x0123456789ABCDEFu64;
        /// let m = 0xEFCDAB8967452301u64;
        ///
        /// assert_eq!(n.swap_bytes(), m);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn swap_bytes(self) -> $T {
            unsafe { $bswap(self as $ActualT) as $T }
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn from_be(x: $T) -> $T {
            if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
        }

        /// Converts an integer from little endian to the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn from_le(x: $T) -> $T {
            if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
        }

        /// Converts `self` to big endian from the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn to_be(self) -> $T { // or not to be?
            if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
        }

        /// Converts `self` to little endian from the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
        ///
        /// # Examples
        ///
        /// ```rust
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
        pub fn to_le(self) -> $T {
            if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
        }

        /// Checked integer addition. Computes `self + other`, returning `None`
        /// if overflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!(5u16.checked_add(65530), Some(65535));
        /// assert_eq!(6u16.checked_add(65530), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_add(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $add_with_overflow, self, other)
        }

        /// Checked integer subtraction. Computes `self - other`, returning
        /// `None` if underflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!((-127i8).checked_sub(1), Some(-128));
        /// assert_eq!((-128i8).checked_sub(1), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_sub(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $sub_with_overflow, self, other)
        }

        /// Checked integer multiplication. Computes `self * other`, returning
        /// `None` if underflow or overflow occurred.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!(5u8.checked_mul(51), Some(255));
        /// assert_eq!(5u8.checked_mul(52), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_mul(self, other: $T) -> Option<$T> {
            checked_op!($T, $ActualT, $mul_with_overflow, self, other)
        }

        /// Checked integer division. Computes `self / other`, returning `None`
        /// if `other == 0` or the operation results in underflow or overflow.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!((-127i8).checked_div(-1), Some(127));
        /// assert_eq!((-128i8).checked_div(-1), None);
        /// assert_eq!((1i8).checked_div(0), None);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn checked_div(self, v: $T) -> Option<$T> {
            match v {
                0 => None,
                v => Some(self / v),
            }
        }

        /// Saturating integer addition. Computes `self + other`, saturating at
        /// the numeric bounds instead of overflowing.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_add(self, other: $T) -> $T {
            match self.checked_add(other) {
                Some(x)                       => x,
                None if other >= <$T as Zero>::zero() => <$T>::max_value(),
                None => <$T>::min_value(),
            }
        }

        /// Saturating integer subtraction. Computes `self - other`, saturating
        /// at the numeric bounds instead of overflowing.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn saturating_sub(self, other: $T) -> $T {
            match self.checked_sub(other) {
                Some(x)                       => x,
                None if other >= <$T as Zero>::zero() => <$T>::min_value(),
                None => <$T>::max_value(),
            }
        }

        /// Wrapping (modular) addition. Computes `self + other`,
        /// wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_add(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_add(self, rhs)
            }
        }

        /// Wrapping (modular) subtraction. Computes `self - other`,
        /// wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_sub(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_sub(self, rhs)
            }
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// other`, wrapping around at the boundary of the type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn wrapping_mul(self, rhs: $T) -> $T {
            unsafe {
                intrinsics::overflowing_mul(self, rhs)
            }
        }

        /// Wrapping (modular) division. Computes `floor(self / other)`,
        /// wrapping around at the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one
        /// divides `MIN / -1` on a signed type (where `MIN` is the
        /// negative minimal value for the type); this is equivalent
        /// to `-MIN`, a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN`
        /// itself..
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_div(self, rhs: $T) -> $T {
            self.overflowing_div(rhs).0
        }

        /// Wrapping (modular) remainder. Computes `self % other`,
        /// wrapping around at the boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically;
        /// implementation artifacts make `x % y` illegal for `MIN /
        /// -1` on a signed type illegal (where `MIN` is the negative
        /// minimal value). In such a case, this function returns `0`.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_rem(self, rhs: $T) -> $T {
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
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_neg(self) -> $T {
            self.overflowing_neg().0
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_shl(self, rhs: u32) -> $T {
            self.overflowing_shl(rhs).0
        }

        /// Panic-free bitwise shift-left; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        #[unstable(feature = "core", since = "1.0.0")]
        #[inline(always)]
        pub fn wrapping_shr(self, rhs: u32) -> $T {
            self.overflowing_shr(rhs).0
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// ```rust
        /// assert_eq!(2i32.pow(4), 16);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn pow(self, mut exp: u32) -> $T {
            let mut base = self;
            let mut acc = <$T as One>::one();

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

        /// Returns `true` iff `self == 2^k` for some `k`.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn is_power_of_two(self) -> bool {
            (self.wrapping_sub(<$T as One>::one())) & self == <$T as Zero>::zero() &&
                !(self == <$T as Zero>::zero())
        }

        /// Returns the smallest power of two greater than or equal to `self`.
        /// Unspecified behavior on overflow.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline]
        pub fn next_power_of_two(self) -> $T {
            let bits = size_of::<$T>() * 8;
            let one: $T = <$T as One>::one();
            one << ((bits - self.wrapping_sub(one).leading_zeros() as usize) % bits)
        }

        /// Returns the smallest power of two greater than or equal to `n`. If
        /// the next power of two is greater than the type's maximum value,
        /// `None` is returned, otherwise the power of two is wrapped in `Some`.
        #[stable(feature = "rust1", since = "1.0.0")]
        pub fn checked_next_power_of_two(self) -> Option<$T> {
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
    uint_impl! { u8 = u8, 8,
        intrinsics::ctpop8,
        intrinsics::ctlz8,
        intrinsics::cttz8,
        bswap8,
        intrinsics::u8_add_with_overflow,
        intrinsics::u8_sub_with_overflow,
        intrinsics::u8_mul_with_overflow }
}

#[lang = "u16"]
impl u16 {
    uint_impl! { u16 = u16, 16,
        intrinsics::ctpop16,
        intrinsics::ctlz16,
        intrinsics::cttz16,
        intrinsics::bswap16,
        intrinsics::u16_add_with_overflow,
        intrinsics::u16_sub_with_overflow,
        intrinsics::u16_mul_with_overflow }
}

#[lang = "u32"]
impl u32 {
    uint_impl! { u32 = u32, 32,
        intrinsics::ctpop32,
        intrinsics::ctlz32,
        intrinsics::cttz32,
        intrinsics::bswap32,
        intrinsics::u32_add_with_overflow,
        intrinsics::u32_sub_with_overflow,
        intrinsics::u32_mul_with_overflow }
}


#[lang = "u64"]
impl u64 {
    uint_impl! { u64 = u64, 64,
        intrinsics::ctpop64,
        intrinsics::ctlz64,
        intrinsics::cttz64,
        intrinsics::bswap64,
        intrinsics::u64_add_with_overflow,
        intrinsics::u64_sub_with_overflow,
        intrinsics::u64_mul_with_overflow }
}

#[cfg(target_pointer_width = "32")]
#[lang = "usize"]
impl usize {
    uint_impl! { usize = u32, 32,
        intrinsics::ctpop32,
        intrinsics::ctlz32,
        intrinsics::cttz32,
        intrinsics::bswap32,
        intrinsics::u32_add_with_overflow,
        intrinsics::u32_sub_with_overflow,
        intrinsics::u32_mul_with_overflow }
}

#[cfg(target_pointer_width = "64")]
#[lang = "usize"]
impl usize {
    uint_impl! { usize = u64, 64,
        intrinsics::ctpop64,
        intrinsics::ctlz64,
        intrinsics::cttz64,
        intrinsics::bswap64,
        intrinsics::u64_add_with_overflow,
        intrinsics::u64_sub_with_overflow,
        intrinsics::u64_mul_with_overflow }
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
pub trait Float {
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

    /// Return the largest integer less than or equal to a number.
    fn floor(self) -> Self;
    /// Return the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self;
    /// Return the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    fn round(self) -> Self;
    /// Return the integer part of a number.
    fn trunc(self) -> Self;
    /// Return the fractional part of a number.
    fn fract(self) -> Self;

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

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Take the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self;

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    fn powi(self, n: i32) -> Self;
    /// Raise a number to a floating point power.
    fn powf(self, n: Self) -> Self;

    /// Take the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    fn sqrt(self) -> Self;
    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    fn rsqrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self;
    /// Returns 2 raised to the power of the number, `2^(self)`.
    fn exp2(self) -> Self;
    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self;
    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(self, base: Self) -> Self;
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self;
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self;

    /// Convert radians to degrees.
    fn to_degrees(self) -> Self;
    /// Convert degrees to radians.
    fn to_radians(self) -> Self;
}

macro_rules! from_str_float_impl {
    ($T:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl FromStr for $T {
            type Err = ParseFloatError;

            /// Converts a string in base 10 to a float.
            /// Accepts an optional decimal exponent.
            ///
            /// This function accepts strings such as
            ///
            /// * '3.14'
            /// * '+3.14', equivalent to '3.14'
            /// * '-3.14'
            /// * '2.5E10', or equivalently, '2.5e10'
            /// * '2.5E-10'
            /// * '.' (understood as 0)
            /// * '5.'
            /// * '.5', or, equivalently,  '0.5'
            /// * '+inf', 'inf', '-inf', 'NaN'
            ///
            /// Leading and trailing whitespace represent an error.
            ///
            /// # Arguments
            ///
            /// * src - A string
            ///
            /// # Return value
            ///
            /// `Err(ParseFloatError)` if the string did not represent a valid
            /// number.  Otherwise, `Ok(n)` where `n` is the floating-point
            /// number represented by `src`.
            #[inline]
            #[allow(deprecated)]
            fn from_str(src: &str) -> Result<$T, ParseFloatError> {
                $T::from_str_radix(src, 10)
            }
        }
    }
}
from_str_float_impl!(f32);
from_str_float_impl!(f64);

macro_rules! from_str_radix_int_impl {
    ($($T:ident)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        #[allow(deprecated)]
        impl FromStr for $T {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<$T, ParseIntError> {
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
    ($($t:ident)*) => ($(impl FromStrRadixHelper for $t {
        fn min_value() -> Self { <$t>::min_value() }
        fn from_u32(u: u32) -> Self { u as $t }
        fn checked_mul(&self, other: u32) -> Option<Self> {
            <$t>::checked_mul(*self, other as $t)
        }
        fn checked_sub(&self, other: u32) -> Option<Self> {
            <$t>::checked_sub(*self, other as $t)
        }
        fn checked_add(&self, other: u32) -> Option<Self> {
            <$t>::checked_add(*self, other as $t)
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

    let is_signed_ty = T::from_u32(0) > T::min_value();

    match src.slice_shift_char() {
        Some(('-', "")) => Err(PIE { kind: Empty }),
        Some(('-', src)) if is_signed_ty => {
            // The number is negative
            let mut result = T::from_u32(0);
            for c in src.chars() {
                let x = match c.to_digit(radix) {
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
            Ok(result)
        },
        Some((_, _)) => {
            // The number is signed
            let mut result = T::from_u32(0);
            for c in src.chars() {
                let x = match c.to_digit(radix) {
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
            Ok(result)
        },
        None => Err(ParseIntError { kind: Empty }),
    }
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
    #[unstable(feature = "core", reason = "available through Error trait")]
    pub fn description(&self) -> &str {
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
        self.description().fmt(f)
    }
}

/// An error which can be returned when parsing a float.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseFloatError { pub kind: FloatErrorKind }

#[derive(Debug, Clone, PartialEq)]
pub enum FloatErrorKind {
    Empty,
    Invalid,
}

impl ParseFloatError {
    #[unstable(feature = "core", reason = "available through Error trait")]
    pub fn description(&self) -> &str {
        match self.kind {
            FloatErrorKind::Empty => "cannot parse float from empty string",
            FloatErrorKind::Invalid => "invalid float literal",
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}
