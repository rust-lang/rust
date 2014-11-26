// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! Numeric traits and functions for the built-in numeric types.

#![stable]
#![allow(missing_docs)]

pub use self::FPCategory::*;

use {int, i8, i16, i32, i64};
use {uint, u8, u16, u32, u64};
use {f32, f64};
use char::Char;
use clone::Clone;
use cmp::{PartialEq, Eq};
use cmp::{PartialOrd, Ord};
use intrinsics;
use iter::IteratorExt;
use kinds::Copy;
use mem::size_of;
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};
use str::{FromStr, from_str, StrPrelude};

/// Simultaneous division and remainder
#[inline]
#[deprecated = "use division and remainder directly"]
pub fn div_rem<T: Div<T, T> + Rem<T, T>>(x: T, y: T) -> (T, T) {
    (x / y, x % y)
}

/// Raises a `base` to the power of `exp`, using exponentiation by squaring.
#[inline]
#[deprecated = "Use Int::pow() instead, as in 2i.pow(4)"]
pub fn pow<T: Int>(base: T, exp: uint) -> T {
    base.pow(exp)
}

/// A built-in signed or unsigned integer.
#[unstable = "recently settled as part of numerics reform"]
pub trait Int
    : Copy + Clone
    + NumCast
    + PartialOrd + Ord
    + PartialEq + Eq
    + Add<Self,Self>
    + Sub<Self,Self>
    + Mul<Self,Self>
    + Div<Self,Self>
    + Rem<Self,Self>
    + Not<Self>
    + BitAnd<Self,Self>
    + BitOr<Self,Self>
    + BitXor<Self,Self>
    + Shl<uint,Self>
    + Shr<uint,Self>
{
    /// Returns the `0` value of this integer type.
    // FIXME (#5527): Should be an associated constant
    fn zero() -> Self;

    /// Returns the `1` value of this integer type.
    // FIXME (#5527): Should be an associated constant
    fn one() -> Self;

    /// Returns the smallest value that can be represented by this integer type.
    // FIXME (#5527): Should be and associated constant
    fn min_value() -> Self;

    /// Returns the largest value that can be represented by this integer type.
    // FIXME (#5527): Should be and associated constant
    fn max_value() -> Self;

    /// Returns the number of ones in the binary representation of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    fn count_ones(self) -> uint;

    /// Returns the number of zeros in the binary representation of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_zeros(), 5);
    /// ```
    #[inline]
    fn count_zeros(self) -> uint {
        (!self).count_ones()
    }

    /// Returns the number of leading zeros in the binary representation
    /// of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.leading_zeros(), 10);
    /// ```
    fn leading_zeros(self) -> uint;

    /// Returns the number of trailing zeros in the binary representation
    /// of `self`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    fn trailing_zeros(self) -> uint;

    /// Shifts the bits to the left by a specified amount amount, `n`, wrapping
    /// the truncated bits to the end of the resulting integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    /// let m = 0x3456789ABCDEF012u64;
    ///
    /// assert_eq!(n.rotate_left(12), m);
    /// ```
    fn rotate_left(self, n: uint) -> Self;

    /// Shifts the bits to the right by a specified amount amount, `n`, wrapping
    /// the truncated bits to the beginning of the resulting integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    /// let m = 0xDEF0123456789ABCu64;
    ///
    /// assert_eq!(n.rotate_right(12), m);
    /// ```
    fn rotate_right(self, n: uint) -> Self;

    /// Reverses the byte order of the integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    /// let m = 0xEFCDAB8967452301u64;
    ///
    /// assert_eq!(n.swap_bytes(), m);
    /// ```
    fn swap_bytes(self) -> Self;

    /// Convert an integer from big endian to the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(Int::from_be(n), n)
    /// } else {
    ///     assert_eq!(Int::from_be(n), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn from_be(x: Self) -> Self {
        if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
    }

    /// Convert an integer from little endian to the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(Int::from_le(n), n)
    /// } else {
    ///     assert_eq!(Int::from_le(n), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn from_le(x: Self) -> Self {
        if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
    }

    /// Convert `self` to big endian from the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(n.to_be(), n)
    /// } else {
    ///     assert_eq!(n.to_be(), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn to_be(self) -> Self { // or not to be?
        if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
    }

    /// Convert `self` to little endian from the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(n.to_le(), n)
    /// } else {
    ///     assert_eq!(n.to_le(), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn to_le(self) -> Self {
        if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
    }

    /// Checked integer addition. Computes `self + other`, returning `None` if
    /// overflow occurred.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// assert_eq!(5u16.checked_add(65530), Some(65535));
    /// assert_eq!(6u16.checked_add(65530), None);
    /// ```
    fn checked_add(self, other: Self) -> Option<Self>;

    /// Checked integer subtraction. Computes `self + other`, returning `None`
    /// if underflow occurred.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// assert_eq!((-127i8).checked_sub(1), Some(-128));
    /// assert_eq!((-128i8).checked_sub(1), None);
    /// ```
    fn checked_sub(self, other: Self) -> Option<Self>;

    /// Checked integer multiplication. Computes `self + other`, returning
    /// `None` if underflow or overflow occurred.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// assert_eq!(5u8.checked_mul(51), Some(255));
    /// assert_eq!(5u8.checked_mul(52), None);
    /// ```
    fn checked_mul(self, other: Self) -> Option<Self>;

    /// Checked integer division. Computes `self + other` returning `None` if
    /// `self == 0` or the operation results in underflow or overflow.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// assert_eq!((-127i8).checked_div(-1), Some(127));
    /// assert_eq!((-128i8).checked_div(-1), None);
    /// assert_eq!((1i8).checked_div(0), None);
    /// ```
    #[inline]
    fn checked_div(self, other: Self) -> Option<Self>;

    /// Saturating integer addition. Computes `self + other`, saturating at
    /// the numeric bounds instead of overflowing.
    #[inline]
    fn saturating_add(self, other: Self) -> Self {
        match self.checked_add(other) {
            Some(x)                      => x,
            None if other >= Int::zero() => Int::max_value(),
            None                         => Int::min_value(),
        }
    }

    /// Saturating integer subtraction. Computes `self - other`, saturating at
    /// the numeric bounds instead of overflowing.
    #[inline]
    fn saturating_sub(self, other: Self) -> Self {
        match self.checked_sub(other) {
            Some(x)                      => x,
            None if other >= Int::zero() => Int::min_value(),
            None                         => Int::max_value(),
        }
    }

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Int;
    ///
    /// assert_eq!(2i.pow(4), 16);
    /// ```
    #[inline]
    fn pow(self, mut exp: uint) -> Self {
        let mut base = self;
        let mut acc: Self = Int::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = base * base;
            exp /= 2;
        }
        acc
    }
}

macro_rules! checked_op {
    ($T:ty, $U:ty, $op:path, $x:expr, $y:expr) => {{
        let (result, overflowed) = unsafe { $op($x as $U, $y as $U) };
        if overflowed { None } else { Some(result as $T) }
    }}
}

macro_rules! uint_impl {
    ($T:ty = $ActualT:ty, $BITS:expr,
     $ctpop:path,
     $ctlz:path,
     $cttz:path,
     $bswap:path,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        #[unstable = "trait is unstable"]
        impl Int for $T {
            #[inline]
            fn zero() -> $T { 0 }

            #[inline]
            fn one() -> $T { 1 }

            #[inline]
            fn min_value() -> $T { 0 }

            #[inline]
            fn max_value() -> $T { -1 }

            #[inline]
            fn count_ones(self) -> uint { unsafe { $ctpop(self as $ActualT) as uint } }

            #[inline]
            fn leading_zeros(self) -> uint { unsafe { $ctlz(self as $ActualT) as uint } }

            #[inline]
            fn trailing_zeros(self) -> uint { unsafe { $cttz(self as $ActualT) as uint } }

            #[inline]
            fn rotate_left(self, n: uint) -> $T {
                // Protect against undefined behaviour for over-long bit shifts
                let n = n % $BITS;
                (self << n) | (self >> (($BITS - n) % $BITS))
            }

            #[inline]
            fn rotate_right(self, n: uint) -> $T {
                // Protect against undefined behaviour for over-long bit shifts
                let n = n % $BITS;
                (self >> n) | (self << (($BITS - n) % $BITS))
            }

            #[inline]
            fn swap_bytes(self) -> $T { unsafe { $bswap(self as $ActualT) as $T } }

            #[inline]
            fn checked_add(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $add_with_overflow, self, other)
            }

            #[inline]
            fn checked_sub(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $sub_with_overflow, self, other)
            }

            #[inline]
            fn checked_mul(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $mul_with_overflow, self, other)
            }

            #[inline]
            fn checked_div(self, v: $T) -> Option<$T> {
                match v {
                    0 => None,
                    v => Some(self / v),
                }
            }
        }
    }
}

/// Swapping a single byte is a no-op. This is marked as `unsafe` for
/// consistency with the other `bswap` intrinsics.
unsafe fn bswap8(x: u8) -> u8 { x }

uint_impl!(u8 = u8, 8,
    intrinsics::ctpop8,
    intrinsics::ctlz8,
    intrinsics::cttz8,
    bswap8,
    intrinsics::u8_add_with_overflow,
    intrinsics::u8_sub_with_overflow,
    intrinsics::u8_mul_with_overflow)

uint_impl!(u16 = u16, 16,
    intrinsics::ctpop16,
    intrinsics::ctlz16,
    intrinsics::cttz16,
    intrinsics::bswap16,
    intrinsics::u16_add_with_overflow,
    intrinsics::u16_sub_with_overflow,
    intrinsics::u16_mul_with_overflow)

uint_impl!(u32 = u32, 32,
    intrinsics::ctpop32,
    intrinsics::ctlz32,
    intrinsics::cttz32,
    intrinsics::bswap32,
    intrinsics::u32_add_with_overflow,
    intrinsics::u32_sub_with_overflow,
    intrinsics::u32_mul_with_overflow)

uint_impl!(u64 = u64, 64,
    intrinsics::ctpop64,
    intrinsics::ctlz64,
    intrinsics::cttz64,
    intrinsics::bswap64,
    intrinsics::u64_add_with_overflow,
    intrinsics::u64_sub_with_overflow,
    intrinsics::u64_mul_with_overflow)

#[cfg(target_word_size = "32")]
uint_impl!(uint = u32, 32,
    intrinsics::ctpop32,
    intrinsics::ctlz32,
    intrinsics::cttz32,
    intrinsics::bswap32,
    intrinsics::u32_add_with_overflow,
    intrinsics::u32_sub_with_overflow,
    intrinsics::u32_mul_with_overflow)

#[cfg(target_word_size = "64")]
uint_impl!(uint = u64, 64,
    intrinsics::ctpop64,
    intrinsics::ctlz64,
    intrinsics::cttz64,
    intrinsics::bswap64,
    intrinsics::u64_add_with_overflow,
    intrinsics::u64_sub_with_overflow,
    intrinsics::u64_mul_with_overflow)

macro_rules! int_impl {
    ($T:ty = $ActualT:ty, $UnsignedT:ty, $BITS:expr,
     $add_with_overflow:path,
     $sub_with_overflow:path,
     $mul_with_overflow:path) => {
        #[unstable = "trait is unstable"]
        impl Int for $T {
            #[inline]
            fn zero() -> $T { 0 }

            #[inline]
            fn one() -> $T { 1 }

            #[inline]
            fn min_value() -> $T { (-1 as $T) << ($BITS - 1) }

            #[inline]
            fn max_value() -> $T { let min: $T = Int::min_value(); !min }

            #[inline]
            fn count_ones(self) -> uint { (self as $UnsignedT).count_ones() }

            #[inline]
            fn leading_zeros(self) -> uint { (self as $UnsignedT).leading_zeros() }

            #[inline]
            fn trailing_zeros(self) -> uint { (self as $UnsignedT).trailing_zeros() }

            #[inline]
            fn rotate_left(self, n: uint) -> $T { (self as $UnsignedT).rotate_left(n) as $T }

            #[inline]
            fn rotate_right(self, n: uint) -> $T { (self as $UnsignedT).rotate_right(n) as $T }

            #[inline]
            fn swap_bytes(self) -> $T { (self as $UnsignedT).swap_bytes() as $T }

            #[inline]
            fn checked_add(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $add_with_overflow, self, other)
            }

            #[inline]
            fn checked_sub(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $sub_with_overflow, self, other)
            }

            #[inline]
            fn checked_mul(self, other: $T) -> Option<$T> {
                checked_op!($T, $ActualT, $mul_with_overflow, self, other)
            }

            #[inline]
            fn checked_div(self, v: $T) -> Option<$T> {
                match v {
                    0   => None,
                   -1 if self == Int::min_value()
                        => None,
                    v   => Some(self / v),
                }
            }
        }
    }
}

int_impl!(i8 = i8, u8, 8,
    intrinsics::i8_add_with_overflow,
    intrinsics::i8_sub_with_overflow,
    intrinsics::i8_mul_with_overflow)

int_impl!(i16 = i16, u16, 16,
    intrinsics::i16_add_with_overflow,
    intrinsics::i16_sub_with_overflow,
    intrinsics::i16_mul_with_overflow)

int_impl!(i32 = i32, u32, 32,
    intrinsics::i32_add_with_overflow,
    intrinsics::i32_sub_with_overflow,
    intrinsics::i32_mul_with_overflow)

int_impl!(i64 = i64, u64, 64,
    intrinsics::i64_add_with_overflow,
    intrinsics::i64_sub_with_overflow,
    intrinsics::i64_mul_with_overflow)

#[cfg(target_word_size = "32")]
int_impl!(int = i32, u32, 32,
    intrinsics::i32_add_with_overflow,
    intrinsics::i32_sub_with_overflow,
    intrinsics::i32_mul_with_overflow)

#[cfg(target_word_size = "64")]
int_impl!(int = i64, u64, 64,
    intrinsics::i64_add_with_overflow,
    intrinsics::i64_sub_with_overflow,
    intrinsics::i64_mul_with_overflow)

/// A built-in two's complement integer.
#[unstable = "recently settled as part of numerics reform"]
pub trait SignedInt
    : Int
    + Neg<Self>
{
    /// Computes the absolute value of `self`. `Int::min_value()` will be
    /// returned if the number is `Int::min_value()`.
    fn abs(self) -> Self;

    /// Returns a number representing sign of `self`.
    ///
    /// - `0` if the number is zero
    /// - `1` if the number is positive
    /// - `-1` if the number is negative
    fn signum(self) -> Self;

    /// Returns `true` if `self` is positive and `false` if the number
    /// is zero or negative.
    fn is_positive(self) -> bool;

    /// Returns `true` if `self` is negative and `false` if the number
    /// is zero or positive.
    fn is_negative(self) -> bool;
}

macro_rules! signed_int_impl {
    ($T:ty) => {
        impl SignedInt for $T {
            #[inline]
            fn abs(self) -> $T {
                if self.is_negative() { -self } else { self }
            }

            #[inline]
            fn signum(self) -> $T {
                match self {
                    n if n > 0 =>  1,
                    0          =>  0,
                    _          => -1,
                }
            }

            #[inline]
            fn is_positive(self) -> bool { self > 0 }

            #[inline]
            fn is_negative(self) -> bool { self < 0 }
        }
    }
}

signed_int_impl!(i8)
signed_int_impl!(i16)
signed_int_impl!(i32)
signed_int_impl!(i64)
signed_int_impl!(int)

/// A built-in unsigned integer.
#[unstable = "recently settled as part of numerics reform"]
pub trait UnsignedInt: Int {
    /// Returns `true` iff `self == 2^k` for some `k`.
    fn is_power_of_two(self) -> bool {
        (self - Int::one()) & self == Int::zero()
    }

    /// Returns the smallest power of two greater than or equal to `self`.
    #[inline]
    fn next_power_of_two(self) -> Self {
        let halfbits = size_of::<Self>() * 4;
        let mut tmp = self - Int::one();
        let mut shift = 1u;
        while shift <= halfbits {
            tmp = tmp | (tmp >> shift);
            shift = shift << 1u;
        }
        tmp + Int::one()
    }

    /// Returns the smallest power of two greater than or equal to `n`. If the
    /// next power of two is greater than the type's maximum value, `None` is
    /// returned, otherwise the power of two is wrapped in `Some`.
    fn checked_next_power_of_two(self) -> Option<Self> {
        let halfbits = size_of::<Self>() * 4;
        let mut tmp = self - Int::one();
        let mut shift = 1u;
        while shift <= halfbits {
            tmp = tmp | (tmp >> shift);
            shift = shift << 1u;
        }
        tmp.checked_add(Int::one())
    }
}

#[unstable = "trait is unstable"]
impl UnsignedInt for uint {}

#[unstable = "trait is unstable"]
impl UnsignedInt for u8 {}

#[unstable = "trait is unstable"]
impl UnsignedInt for u16 {}

#[unstable = "trait is unstable"]
impl UnsignedInt for u32 {}

#[unstable = "trait is unstable"]
impl UnsignedInt for u64 {}

/// A generic trait for converting a value to a number.
#[experimental = "trait is likely to be removed"]
pub trait ToPrimitive {
    /// Converts the value of `self` to an `int`.
    #[inline]
    fn to_int(&self) -> Option<int> {
        self.to_i64().and_then(|x| x.to_int())
    }

    /// Converts the value of `self` to an `i8`.
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.to_i64().and_then(|x| x.to_i8())
    }

    /// Converts the value of `self` to an `i16`.
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.to_i64().and_then(|x| x.to_i16())
    }

    /// Converts the value of `self` to an `i32`.
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.to_i64().and_then(|x| x.to_i32())
    }

    /// Converts the value of `self` to an `i64`.
    fn to_i64(&self) -> Option<i64>;

    /// Converts the value of `self` to an `uint`.
    #[inline]
    fn to_uint(&self) -> Option<uint> {
        self.to_u64().and_then(|x| x.to_uint())
    }

    /// Converts the value of `self` to an `u8`.
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.to_u64().and_then(|x| x.to_u8())
    }

    /// Converts the value of `self` to an `u16`.
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.to_u64().and_then(|x| x.to_u16())
    }

    /// Converts the value of `self` to an `u32`.
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.to_u64().and_then(|x| x.to_u32())
    }

    /// Converts the value of `self` to an `u64`.
    #[inline]
    fn to_u64(&self) -> Option<u64>;

    /// Converts the value of `self` to an `f32`.
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f64().and_then(|x| x.to_f32())
    }

    /// Converts the value of `self` to an `f64`.
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.to_i64().and_then(|x| x.to_f64())
    }
}

macro_rules! impl_to_primitive_int_to_int(
    ($SrcT:ty, $DstT:ty, $slf:expr) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some($slf as $DstT)
            } else {
                let n = $slf as i64;
                let min_value: $DstT = Int::min_value();
                let max_value: $DstT = Int::max_value();
                if min_value as i64 <= n && n <= max_value as i64 {
                    Some($slf as $DstT)
                } else {
                    None
                }
            }
        }
    )
)

macro_rules! impl_to_primitive_int_to_uint(
    ($SrcT:ty, $DstT:ty, $slf:expr) => (
        {
            let zero: $SrcT = Int::zero();
            let max_value: $DstT = Int::max_value();
            if zero <= $slf && $slf as u64 <= max_value as u64 {
                Some($slf as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_int(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { impl_to_primitive_int_to_int!($T, int, *self) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_int_to_int!($T, i8, *self) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_int_to_int!($T, i16, *self) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_int_to_int!($T, i32, *self) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_int_to_int!($T, i64, *self) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_int_to_uint!($T, uint, *self) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_int_to_uint!($T, u8, *self) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_int_to_uint!($T, u16, *self) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_int_to_uint!($T, u32, *self) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_int_to_uint!($T, u64, *self) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { Some(*self as f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )
)

impl_to_primitive_int!(int)
impl_to_primitive_int!(i8)
impl_to_primitive_int!(i16)
impl_to_primitive_int!(i32)
impl_to_primitive_int!(i64)

macro_rules! impl_to_primitive_uint_to_int(
    ($DstT:ty, $slf:expr) => (
        {
            let max_value: $DstT = Int::max_value();
            if $slf as u64 <= max_value as u64 {
                Some($slf as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_uint_to_uint(
    ($SrcT:ty, $DstT:ty, $slf:expr) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some($slf as $DstT)
            } else {
                let zero: $SrcT = Int::zero();
                let max_value: $DstT = Int::max_value();
                if zero <= $slf && $slf as u64 <= max_value as u64 {
                    Some($slf as $DstT)
                } else {
                    None
                }
            }
        }
    )
)

macro_rules! impl_to_primitive_uint(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { impl_to_primitive_uint_to_int!(int, *self) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_uint_to_int!(i8, *self) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_uint_to_int!(i16, *self) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_uint_to_int!(i32, *self) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_uint_to_int!(i64, *self) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_uint_to_uint!($T, uint, *self) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_uint_to_uint!($T, u8, *self) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_uint_to_uint!($T, u16, *self) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_uint_to_uint!($T, u32, *self) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_uint_to_uint!($T, u64, *self) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { Some(*self as f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { Some(*self as f64) }
        }
    )
)

impl_to_primitive_uint!(uint)
impl_to_primitive_uint!(u8)
impl_to_primitive_uint!(u16)
impl_to_primitive_uint!(u32)
impl_to_primitive_uint!(u64)

macro_rules! impl_to_primitive_float_to_float(
    ($SrcT:ty, $DstT:ty, $slf:expr) => (
        if size_of::<$SrcT>() <= size_of::<$DstT>() {
            Some($slf as $DstT)
        } else {
            let n = $slf as f64;
            let max_value: $SrcT = Float::max_value();
            if -max_value as f64 <= n && n <= max_value as f64 {
                Some($slf as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_float(
    ($T:ty) => (
        impl ToPrimitive for $T {
            #[inline]
            fn to_int(&self) -> Option<int> { Some(*self as int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { Some(*self as i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { Some(*self as i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { Some(*self as i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { Some(*self as i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { Some(*self as uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { Some(*self as u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { Some(*self as u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { Some(*self as u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { Some(*self as u64) }

            #[inline]
            fn to_f32(&self) -> Option<f32> { impl_to_primitive_float_to_float!($T, f32, *self) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { impl_to_primitive_float_to_float!($T, f64, *self) }
        }
    )
)

impl_to_primitive_float!(f32)
impl_to_primitive_float!(f64)

/// A generic trait for converting a number to a value.
#[experimental = "trait is likely to be removed"]
pub trait FromPrimitive {
    /// Convert an `int` to return an optional value of this type. If the
    /// value cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_int(n: int) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i8` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i16` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }

    /// Convert an `i64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    fn from_i64(n: i64) -> Option<Self>;

    /// Convert an `uint` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_uint(n: uint) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u8` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u16` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        FromPrimitive::from_u64(n as u64)
    }

    /// Convert an `u64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    fn from_u64(n: u64) -> Option<Self>;

    /// Convert a `f32` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        FromPrimitive::from_f64(n as f64)
    }

    /// Convert a `f64` to return an optional value of this type. If the
    /// type cannot be represented by this value, the `None` is returned.
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        FromPrimitive::from_i64(n as i64)
    }
}

/// A utility function that just calls `FromPrimitive::from_int`.
#[experimental = "likely to be removed"]
pub fn from_int<A: FromPrimitive>(n: int) -> Option<A> {
    FromPrimitive::from_int(n)
}

/// A utility function that just calls `FromPrimitive::from_i8`.
#[experimental = "likely to be removed"]
pub fn from_i8<A: FromPrimitive>(n: i8) -> Option<A> {
    FromPrimitive::from_i8(n)
}

/// A utility function that just calls `FromPrimitive::from_i16`.
#[experimental = "likely to be removed"]
pub fn from_i16<A: FromPrimitive>(n: i16) -> Option<A> {
    FromPrimitive::from_i16(n)
}

/// A utility function that just calls `FromPrimitive::from_i32`.
#[experimental = "likely to be removed"]
pub fn from_i32<A: FromPrimitive>(n: i32) -> Option<A> {
    FromPrimitive::from_i32(n)
}

/// A utility function that just calls `FromPrimitive::from_i64`.
#[experimental = "likely to be removed"]
pub fn from_i64<A: FromPrimitive>(n: i64) -> Option<A> {
    FromPrimitive::from_i64(n)
}

/// A utility function that just calls `FromPrimitive::from_uint`.
#[experimental = "likely to be removed"]
pub fn from_uint<A: FromPrimitive>(n: uint) -> Option<A> {
    FromPrimitive::from_uint(n)
}

/// A utility function that just calls `FromPrimitive::from_u8`.
#[experimental = "likely to be removed"]
pub fn from_u8<A: FromPrimitive>(n: u8) -> Option<A> {
    FromPrimitive::from_u8(n)
}

/// A utility function that just calls `FromPrimitive::from_u16`.
#[experimental = "likely to be removed"]
pub fn from_u16<A: FromPrimitive>(n: u16) -> Option<A> {
    FromPrimitive::from_u16(n)
}

/// A utility function that just calls `FromPrimitive::from_u32`.
#[experimental = "likely to be removed"]
pub fn from_u32<A: FromPrimitive>(n: u32) -> Option<A> {
    FromPrimitive::from_u32(n)
}

/// A utility function that just calls `FromPrimitive::from_u64`.
#[experimental = "likely to be removed"]
pub fn from_u64<A: FromPrimitive>(n: u64) -> Option<A> {
    FromPrimitive::from_u64(n)
}

/// A utility function that just calls `FromPrimitive::from_f32`.
#[experimental = "likely to be removed"]
pub fn from_f32<A: FromPrimitive>(n: f32) -> Option<A> {
    FromPrimitive::from_f32(n)
}

/// A utility function that just calls `FromPrimitive::from_f64`.
#[experimental = "likely to be removed"]
pub fn from_f64<A: FromPrimitive>(n: f64) -> Option<A> {
    FromPrimitive::from_f64(n)
}

macro_rules! impl_from_primitive(
    ($T:ty, $to_ty:ident) => (
        impl FromPrimitive for $T {
            #[inline] fn from_int(n: int) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_i8(n: i8) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_i16(n: i16) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_i32(n: i32) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_i64(n: i64) -> Option<$T> { n.$to_ty() }

            #[inline] fn from_uint(n: uint) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_u8(n: u8) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_u16(n: u16) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_u32(n: u32) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_u64(n: u64) -> Option<$T> { n.$to_ty() }

            #[inline] fn from_f32(n: f32) -> Option<$T> { n.$to_ty() }
            #[inline] fn from_f64(n: f64) -> Option<$T> { n.$to_ty() }
        }
    )
)

impl_from_primitive!(int, to_int)
impl_from_primitive!(i8, to_i8)
impl_from_primitive!(i16, to_i16)
impl_from_primitive!(i32, to_i32)
impl_from_primitive!(i64, to_i64)
impl_from_primitive!(uint, to_uint)
impl_from_primitive!(u8, to_u8)
impl_from_primitive!(u16, to_u16)
impl_from_primitive!(u32, to_u32)
impl_from_primitive!(u64, to_u64)
impl_from_primitive!(f32, to_f32)
impl_from_primitive!(f64, to_f64)

/// Cast from one machine scalar to another.
///
/// # Example
///
/// ```
/// use std::num;
///
/// let twenty: f32 = num::cast(0x14i).unwrap();
/// assert_eq!(twenty, 20f32);
/// ```
///
#[inline]
#[experimental = "likely to be removed"]
pub fn cast<T: NumCast,U: NumCast>(n: T) -> Option<U> {
    NumCast::from(n)
}

/// An interface for casting between machine scalars.
#[experimental = "trait is likely to be removed"]
pub trait NumCast: ToPrimitive {
    /// Creates a number from another value that can be converted into a primitive via the
    /// `ToPrimitive` trait.
    fn from<T: ToPrimitive>(n: T) -> Option<Self>;
}

macro_rules! impl_num_cast(
    ($T:ty, $conv:ident) => (
        impl NumCast for $T {
            #[inline]
            fn from<N: ToPrimitive>(n: N) -> Option<$T> {
                // `$conv` could be generated using `concat_idents!`, but that
                // macro seems to be broken at the moment
                n.$conv()
            }
        }
    )
)

impl_num_cast!(u8,    to_u8)
impl_num_cast!(u16,   to_u16)
impl_num_cast!(u32,   to_u32)
impl_num_cast!(u64,   to_u64)
impl_num_cast!(uint,  to_uint)
impl_num_cast!(i8,    to_i8)
impl_num_cast!(i16,   to_i16)
impl_num_cast!(i32,   to_i32)
impl_num_cast!(i64,   to_i64)
impl_num_cast!(int,   to_int)
impl_num_cast!(f32,   to_f32)
impl_num_cast!(f64,   to_f64)

/// Used for representing the classification of floating point numbers
#[deriving(PartialEq, Show)]
#[unstable = "may be renamed"]
pub enum FPCategory {
    /// "Not a Number", often obtained by dividing by zero
    FPNaN,
    /// Positive or negative infinity
    FPInfinite ,
    /// Positive or negative zero
    FPZero,
    /// De-normalized floating point representation (less precise than `FPNormal`)
    FPSubnormal,
    /// A regular floating point number
    FPNormal,
}

/// A built-in floating point number.
// FIXME(#5527): In a future version of Rust, many of these functions will
//               become constants.
//
// FIXME(#8888): Several of these functions have a parameter named
//               `unused_self`. Removing it requires #8888 to be fixed.
#[unstable = "recently settled as part of numerics reform"]
pub trait Float
    : Copy + Clone
    + NumCast
    + PartialOrd
    + PartialEq
    + Neg<Self>
    + Add<Self,Self>
    + Sub<Self,Self>
    + Mul<Self,Self>
    + Div<Self,Self>
    + Rem<Self,Self>
{
    /// Returns the NaN value.
    fn nan() -> Self;
    /// Returns the infinite value.
    fn infinity() -> Self;
    /// Returns the negative infinite value.
    fn neg_infinity() -> Self;
    /// Returns the `0` value.
    fn zero() -> Self;
    /// Returns -0.0.
    fn neg_zero() -> Self;
    /// Returns the `1` value.
    fn one() -> Self;

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
    fn classify(self) -> FPCategory;

    // FIXME (#5527): These should be associated constants

    /// Returns the number of binary digits of mantissa that this type supports.
    fn mantissa_digits(unused_self: Option<Self>) -> uint;
    /// Returns the number of base-10 digits of precision that this type supports.
    fn digits(unused_self: Option<Self>) -> uint;
    /// Returns the difference between 1.0 and the smallest representable number larger than 1.0.
    fn epsilon() -> Self;
    /// Returns the minimum binary exponent that this type can represent.
    fn min_exp(unused_self: Option<Self>) -> int;
    /// Returns the maximum binary exponent that this type can represent.
    fn max_exp(unused_self: Option<Self>) -> int;
    /// Returns the minimum base-10 exponent that this type can represent.
    fn min_10_exp(unused_self: Option<Self>) -> int;
    /// Returns the maximum base-10 exponent that this type can represent.
    fn max_10_exp(unused_self: Option<Self>) -> int;
    /// Returns the smallest finite value that this type can represent.
    fn min_value() -> Self;
    /// Returns the smallest normalized positive number that this type can represent.
    fn min_pos_value(unused_self: Option<Self>) -> Self;
    /// Returns the largest finite value that this type can represent.
    fn max_value() -> Self;

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

    /// sqrt(2.0).
    fn sqrt2() -> Self;
    /// 1.0 / sqrt(2.0).
    fn frac_1_sqrt2() -> Self;

    /// Take the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    fn sqrt(self) -> Self;
    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    fn rsqrt(self) -> Self;

    /// Archimedes' constant.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn pi() -> Self;
    /// 2.0 * pi.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn two_pi() -> Self;
    /// pi / 2.0.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_pi_2() -> Self;
    /// pi / 3.0.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_pi_3() -> Self;
    /// pi / 4.0.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_pi_4() -> Self;
    /// pi / 6.0.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_pi_6() -> Self;
    /// pi / 8.0.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_pi_8() -> Self;
    /// 1.0 / pi.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_1_pi() -> Self;
    /// 2.0 / pi.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_2_pi() -> Self;
    /// 2.0 / sqrt(pi).
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn frac_2_sqrtpi() -> Self;

    /// Euler's number.
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn e() -> Self;
    /// log2(e).
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn log2_e() -> Self;
    /// log10(e).
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn log10_e() -> Self;
    /// ln(2.0).
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn ln_2() -> Self;
    /// ln(10.0).
    #[deprecated = "use f32::consts or f64::consts instead"]
    fn ln_10() -> Self;

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

/// A generic trait for converting a string with a radix (base) to a value
#[experimental = "might need to return Result"]
pub trait FromStrRadix {
    fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
}

/// A utility function that just calls FromStrRadix::from_str_radix.
#[experimental = "might need to return Result"]
pub fn from_str_radix<T: FromStrRadix>(str: &str, radix: uint) -> Option<T> {
    FromStrRadix::from_str_radix(str, radix)
}

macro_rules! from_str_radix_float_impl {
    ($T:ty) => {
        #[experimental = "might need to return Result"]
        impl FromStr for $T {
            /// Convert a string in base 10 to a float.
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
            /// `None` if the string did not represent a valid number.  Otherwise,
            /// `Some(n)` where `n` is the floating-point number represented by `src`.
            #[inline]
            fn from_str(src: &str) -> Option<$T> {
                from_str_radix(src, 10)
            }
        }

        #[experimental = "might need to return Result"]
        impl FromStrRadix for $T {
            /// Convert a string in a given base to a float.
            ///
            /// Due to possible conflicts, this function does **not** accept
            /// the special values `inf`, `-inf`, `+inf` and `NaN`, **nor**
            /// does it recognize exponents of any kind.
            ///
            /// Leading and trailing whitespace represent an error.
            ///
            /// # Arguments
            ///
            /// * src - A string
            /// * radix - The base to use. Must lie in the range [2 .. 36]
            ///
            /// # Return value
            ///
            /// `None` if the string did not represent a valid number. Otherwise,
            /// `Some(n)` where `n` is the floating-point number represented by `src`.
            fn from_str_radix(src: &str, radix: uint) -> Option<$T> {
               assert!(radix >= 2 && radix <= 36,
                       "from_str_radix_float: must lie in the range `[2, 36]` - found {}",
                       radix);

                // Special values
                match src {
                    "inf"   => return Some(Float::infinity()),
                    "-inf"  => return Some(Float::neg_infinity()),
                    "NaN"   => return Some(Float::nan()),
                    _       => {},
                }

                let (is_positive, src) =  match src.slice_shift_char() {
                    None             => return None,
                    Some(('-', ""))  => return None,
                    Some(('-', src)) => (false, src),
                    Some((_, _))     => (true,  src),
                };

                // The significand to accumulate
                let mut sig = if is_positive { 0.0 } else { -0.0 };
                // Necessary to detect overflow
                let mut prev_sig = sig;
                let mut cs = src.chars().enumerate();
                // Exponent prefix and exponent index offset
                let mut exp_info = None::<(char, uint)>;

                // Parse the integer part of the significand
                for (i, c) in cs {
                    match c.to_digit(radix) {
                        Some(digit) => {
                            // shift significand one digit left
                            sig = sig * (radix as $T);

                            // add/subtract current digit depending on sign
                            if is_positive {
                                sig = sig + ((digit as int) as $T);
                            } else {
                                sig = sig - ((digit as int) as $T);
                            }

                            // Detect overflow by comparing to last value, except
                            // if we've not seen any non-zero digits.
                            if prev_sig != 0.0 {
                                if is_positive && sig <= prev_sig
                                    { return Some(Float::infinity()); }
                                if !is_positive && sig >= prev_sig
                                    { return Some(Float::neg_infinity()); }

                                // Detect overflow by reversing the shift-and-add process
                                if is_positive && (prev_sig != (sig - digit as $T) / radix as $T)
                                    { return Some(Float::infinity()); }
                                if !is_positive && (prev_sig != (sig + digit as $T) / radix as $T)
                                    { return Some(Float::neg_infinity()); }
                            }
                            prev_sig = sig;
                        },
                        None => match c {
                            'e' | 'E' | 'p' | 'P' => {
                                exp_info = Some((c, i + 1));
                                break;  // start of exponent
                            },
                            '.' => {
                                break;  // start of fractional part
                            },
                            _ => {
                                return None;
                            },
                        },
                    }
                }

                // If we are not yet at the exponent parse the fractional
                // part of the significand
                if exp_info.is_none() {
                    let mut power = 1.0;
                    for (i, c) in cs {
                        match c.to_digit(radix) {
                            Some(digit) => {
                                // Decrease power one order of magnitude
                                power = power / (radix as $T);
                                // add/subtract current digit depending on sign
                                sig = if is_positive {
                                    sig + (digit as $T) * power
                                } else {
                                    sig - (digit as $T) * power
                                };
                                // Detect overflow by comparing to last value
                                if is_positive && sig < prev_sig
                                    { return Some(Float::infinity()); }
                                if !is_positive && sig > prev_sig
                                    { return Some(Float::neg_infinity()); }
                                prev_sig = sig;
                            },
                            None => match c {
                                'e' | 'E' | 'p' | 'P' => {
                                    exp_info = Some((c, i + 1));
                                    break; // start of exponent
                                },
                                _ => {
                                    return None; // invalid number
                                },
                            },
                        }
                    }
                }

                // Parse and calculate the exponent
                let exp = match exp_info {
                    Some((c, offset)) => {
                        let base = match c {
                            'E' | 'e' if radix == 10 => 10u as $T,
                            'P' | 'p' if radix == 16 => 2u as $T,
                            _ => return None,
                        };

                        // Parse the exponent as decimal integer
                        let src = src[offset..];
                        let (is_positive, exp) = match src.slice_shift_char() {
                            Some(('-', src)) => (false, from_str::<uint>(src)),
                            Some(('+', src)) => (true,  from_str::<uint>(src)),
                            Some((_, _))     => (true,  from_str::<uint>(src)),
                            None             => return None,
                        };

                        match (is_positive, exp) {
                            (true,  Some(exp)) => base.powi(exp as i32),
                            (false, Some(exp)) => 1.0 / base.powi(exp as i32),
                            (_, None)          => return None,
                        }
                    },
                    None => 1.0, // no exponent
                };

                Some(sig * exp)
            }
        }
    }
}
from_str_radix_float_impl!(f32)
from_str_radix_float_impl!(f64)

macro_rules! from_str_radix_int_impl {
    ($T:ty) => {
        #[experimental = "might need to return Result"]
        impl FromStr for $T {
            #[inline]
            fn from_str(src: &str) -> Option<$T> {
                from_str_radix(src, 10)
            }
        }

        #[experimental = "might need to return Result"]
        impl FromStrRadix for $T {
            fn from_str_radix(src: &str, radix: uint) -> Option<$T> {
                assert!(radix >= 2 && radix <= 36,
                       "from_str_radix_int: must lie in the range `[2, 36]` - found {}",
                       radix);

                let is_signed_ty = (0 as $T) > Int::min_value();

                match src.slice_shift_char() {
                    Some(('-', src)) if is_signed_ty => {
                        // The number is negative
                        let mut result = 0;
                        for c in src.chars() {
                            let x = match c.to_digit(radix) {
                                Some(x) => x,
                                None => return None,
                            };
                            result = match result.checked_mul(radix as $T) {
                                Some(result) => result,
                                None => return None,
                            };
                            result = match result.checked_sub(x as $T) {
                                Some(result) => result,
                                None => return None,
                            };
                        }
                        Some(result)
                    },
                    Some((_, _)) => {
                        // The number is signed
                        let mut result = 0;
                        for c in src.chars() {
                            let x = match c.to_digit(radix) {
                                Some(x) => x,
                                None => return None,
                            };
                            result = match result.checked_mul(radix as $T) {
                                Some(result) => result,
                                None => return None,
                            };
                            result = match result.checked_add(x as $T) {
                                Some(result) => result,
                                None => return None,
                            };
                        }
                        Some(result)
                    },
                    None => None,
                }
            }
        }
    }
}
from_str_radix_int_impl!(int)
from_str_radix_int_impl!(i8)
from_str_radix_int_impl!(i16)
from_str_radix_int_impl!(i32)
from_str_radix_int_impl!(i64)
from_str_radix_int_impl!(uint)
from_str_radix_int_impl!(u8)
from_str_radix_int_impl!(u16)
from_str_radix_int_impl!(u32)
from_str_radix_int_impl!(u64)

// DEPRECATED

macro_rules! trait_impl {
    ($name:ident for $($t:ty)*) => {
        $(#[allow(deprecated)] impl $name for $t {})*
    };
}

#[deprecated = "Generalised numbers are no longer supported"]
#[allow(deprecated)]
pub trait Num: PartialEq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}
trait_impl!(Num for uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

#[deprecated = "Generalised unsigned numbers are no longer supported"]
#[allow(deprecated)]
pub trait Unsigned: Num {}
trait_impl!(Unsigned for uint u8 u16 u32 u64)

#[deprecated = "Use `Float` or `Int`"]
#[allow(deprecated)]
pub trait Primitive: Copy + Clone + Num + NumCast + PartialOrd {}
trait_impl!(Primitive for uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

#[deprecated = "The generic `Zero` trait will be removed soon."]
pub trait Zero: Add<Self, Self> {
    #[deprecated = "Use `Int::zero()` or `Float::zero()`."]
    fn zero() -> Self;
    #[deprecated = "Use `x == Int::zero()` or `x == Float::zero()`."]
    fn is_zero(&self) -> bool;
}
#[deprecated = "Use `Int::zero()` or `Float::zero()`."]
#[allow(deprecated)]
pub fn zero<T: Zero>() -> T { Zero::zero() }
macro_rules! zero_impl {
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            fn zero() -> $t { $v }
            fn is_zero(&self) -> bool { *self == $v }
        }
    }
}
zero_impl!(uint, 0u)
zero_impl!(u8,   0u8)
zero_impl!(u16,  0u16)
zero_impl!(u32,  0u32)
zero_impl!(u64,  0u64)
zero_impl!(int, 0i)
zero_impl!(i8,  0i8)
zero_impl!(i16, 0i16)
zero_impl!(i32, 0i32)
zero_impl!(i64, 0i64)
zero_impl!(f32, 0.0f32)
zero_impl!(f64, 0.0f64)

#[deprecated = "The generic `One` trait will be removed soon."]
pub trait One: Mul<Self, Self> {
    #[deprecated = "Use `Int::one()` or `Float::one()`."]
    fn one() -> Self;
}
#[deprecated = "Use `Int::one()` or `Float::one()`."]
#[allow(deprecated)]
pub fn one<T: One>() -> T { One::one() }
macro_rules! one_impl {
    ($t:ty, $v:expr) => {
        impl One for $t {
            fn one() -> $t { $v }
        }
    }
}
one_impl!(uint, 1u)
one_impl!(u8,  1u8)
one_impl!(u16, 1u16)
one_impl!(u32, 1u32)
one_impl!(u64, 1u64)
one_impl!(int, 1i)
one_impl!(i8,  1i8)
one_impl!(i16, 1i16)
one_impl!(i32, 1i32)
one_impl!(i64, 1i64)
one_impl!(f32, 1.0f32)
one_impl!(f64, 1.0f64)

#[deprecated = "Use `UnsignedInt::next_power_of_two`"]
pub fn next_power_of_two<T: UnsignedInt>(n: T) -> T {
    n.next_power_of_two()
}
#[deprecated = "Use `UnsignedInt::is_power_of_two`"]
pub fn is_power_of_two<T: UnsignedInt>(n: T) -> bool {
    n.is_power_of_two()
}
#[deprecated = "Use `UnsignedInt::checked_next_power_of_two`"]
pub fn checked_next_power_of_two<T: UnsignedInt>(n: T) -> Option<T> {
    n.checked_next_power_of_two()
}

#[deprecated = "Generalised bounded values are no longer supported"]
pub trait Bounded {
    #[deprecated = "Use `Int::min_value` or `Float::min_value`"]
    fn min_value() -> Self;
    #[deprecated = "Use `Int::max_value` or `Float::max_value`"]
    fn max_value() -> Self;
}
macro_rules! bounded_impl {
    ($T:ty, $min:expr, $max:expr) => {
        impl Bounded for $T {
            #[inline]
            fn min_value() -> $T { $min }

            #[inline]
            fn max_value() -> $T { $max }
        }
    };
}
bounded_impl!(uint, uint::MIN, uint::MAX)
bounded_impl!(u8, u8::MIN, u8::MAX)
bounded_impl!(u16, u16::MIN, u16::MAX)
bounded_impl!(u32, u32::MIN, u32::MAX)
bounded_impl!(u64, u64::MIN, u64::MAX)
bounded_impl!(int, int::MIN, int::MAX)
bounded_impl!(i8, i8::MIN, i8::MAX)
bounded_impl!(i16, i16::MIN, i16::MAX)
bounded_impl!(i32, i32::MIN, i32::MAX)
bounded_impl!(i64, i64::MIN, i64::MAX)
bounded_impl!(f32, f32::MIN_VALUE, f32::MAX_VALUE)
bounded_impl!(f64, f64::MIN_VALUE, f64::MAX_VALUE)
