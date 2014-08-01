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

//! Numeric traits and functions for generic mathematics

#![allow(missing_doc)]

use intrinsics;
use {int, i8, i16, i32, i64};
use {uint, u8, u16, u32, u64};
use {f32, f64};
use clone::Clone;
use cmp::{PartialEq, PartialOrd};
use kinds::Copy;
use mem::size_of;
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};

/// The base trait for numeric types
pub trait Num: PartialEq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}

macro_rules! trait_impl(
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {}
    )*)
)

trait_impl!(Num for uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

/// Simultaneous division and remainder
#[inline]
pub fn div_rem<T: Div<T, T> + Rem<T, T>>(x: T, y: T) -> (T, T) {
    (x / y, x % y)
}

/// Defines an additive identity element for `Self`.
///
/// # Deriving
///
/// This trait can be automatically be derived using `#[deriving(Zero)]`
/// attribute. If you choose to use this, make sure that the laws outlined in
/// the documentation for `Zero::zero` still hold.
pub trait Zero: Add<Self, Self> {
    /// Returns the additive identity element of `Self`, `0`.
    ///
    /// # Laws
    ///
    /// ~~~text
    /// a + 0 = a       ∀ a ∈ Self
    /// 0 + a = a       ∀ a ∈ Self
    /// ~~~
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn zero() -> Self;

    /// Returns `true` if `self` is equal to the additive identity.
    #[inline]
    fn is_zero(&self) -> bool;
}

macro_rules! zero_impl(
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> $t { $v }
            #[inline]
            fn is_zero(&self) -> bool { *self == $v }
        }
    }
)

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

/// Returns the additive identity, `0`.
#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

/// Defines a multiplicative identity element for `Self`.
pub trait One: Mul<Self, Self> {
    /// Returns the multiplicative identity element of `Self`, `1`.
    ///
    /// # Laws
    ///
    /// ~~~text
    /// a * 1 = a       ∀ a ∈ Self
    /// 1 * a = a       ∀ a ∈ Self
    /// ~~~
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn one() -> Self;
}

macro_rules! one_impl(
    ($t:ty, $v:expr) => {
        impl One for $t {
            #[inline]
            fn one() -> $t { $v }
        }
    }
)

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

/// Returns the multiplicative identity, `1`.
#[inline(always)] pub fn one<T: One>() -> T { One::one() }

/// Useful functions for signed numbers (i.e. numbers that can be negative).
pub trait Signed: Num + Neg<Self> {
    /// Computes the absolute value.
    ///
    /// For `f32` and `f64`, `NaN` will be returned if the number is `NaN`.
    fn abs(&self) -> Self;

    /// The positive difference of two numbers.
    ///
    /// Returns `zero` if the number is less than or equal to `other`, otherwise the difference
    /// between `self` and `other` is returned.
    fn abs_sub(&self, other: &Self) -> Self;

    /// Returns the sign of the number.
    ///
    /// For `f32` and `f64`:
    ///
    /// * `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// * `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// * `NaN` if the number is `NaN`
    ///
    /// For `int`:
    ///
    /// * `0` if the number is zero
    /// * `1` if the number is positive
    /// * `-1` if the number is negative
    fn signum(&self) -> Self;

    /// Returns true if the number is positive and false if the number is zero or negative.
    fn is_positive(&self) -> bool;

    /// Returns true if the number is negative and false if the number is zero or positive.
    fn is_negative(&self) -> bool;
}

macro_rules! signed_impl(
    ($($t:ty)*) => ($(
        impl Signed for $t {
            #[inline]
            fn abs(&self) -> $t {
                if self.is_negative() { -*self } else { *self }
            }

            #[inline]
            fn abs_sub(&self, other: &$t) -> $t {
                if *self <= *other { 0 } else { *self - *other }
            }

            #[inline]
            fn signum(&self) -> $t {
                match *self {
                    n if n > 0 => 1,
                    0 => 0,
                    _ => -1,
                }
            }

            #[inline]
            fn is_positive(&self) -> bool { *self > 0 }

            #[inline]
            fn is_negative(&self) -> bool { *self < 0 }
        }
    )*)
)

signed_impl!(int i8 i16 i32 i64)

macro_rules! signed_float_impl(
    ($t:ty, $nan:expr, $inf:expr, $neg_inf:expr, $fabs:path, $fcopysign:path, $fdim:ident) => {
        impl Signed for $t {
            /// Computes the absolute value. Returns `NAN` if the number is `NAN`.
            #[inline]
            fn abs(&self) -> $t {
                unsafe { $fabs(*self) }
            }

            /// The positive difference of two numbers. Returns `0.0` if the number is
            /// less than or equal to `other`, otherwise the difference between`self`
            /// and `other` is returned.
            #[inline]
            fn abs_sub(&self, other: &$t) -> $t {
                extern { fn $fdim(a: $t, b: $t) -> $t; }
                unsafe { $fdim(*self, *other) }
            }

            /// # Returns
            ///
            /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
            /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
            /// - `NAN` if the number is NaN
            #[inline]
            fn signum(&self) -> $t {
                if self != self { $nan } else {
                    unsafe { $fcopysign(1.0, *self) }
                }
            }

            /// Returns `true` if the number is positive, including `+0.0` and `INFINITY`
            #[inline]
            fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == $inf }

            /// Returns `true` if the number is negative, including `-0.0` and `NEG_INFINITY`
            #[inline]
            fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == $neg_inf }
        }
    }
)

signed_float_impl!(f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
                   intrinsics::fabsf32, intrinsics::copysignf32, fdimf)
signed_float_impl!(f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
                   intrinsics::fabsf64, intrinsics::copysignf64, fdim)

/// Computes the absolute value.
///
/// For `f32` and `f64`, `NaN` will be returned if the number is `NaN`
#[inline(always)]
pub fn abs<T: Signed>(value: T) -> T {
    value.abs()
}

/// The positive difference of two numbers.
///
/// Returns `zero` if the number is less than or equal to `other`,
/// otherwise the difference between `self` and `other` is returned.
#[inline(always)]
pub fn abs_sub<T: Signed>(x: T, y: T) -> T {
    x.abs_sub(&y)
}

/// Returns the sign of the number.
///
/// For `f32` and `f64`:
///
/// * `1.0` if the number is positive, `+0.0` or `INFINITY`
/// * `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
/// * `NaN` if the number is `NaN`
///
/// For int:
///
/// * `0` if the number is zero
/// * `1` if the number is positive
/// * `-1` if the number is negative
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

/// A trait for values which cannot be negative
pub trait Unsigned: Num {}

trait_impl!(Unsigned for uint u8 u16 u32 u64)

/// Raises a value to the power of exp, using exponentiation by squaring.
///
/// # Example
///
/// ```rust
/// use std::num;
///
/// assert_eq!(num::pow(2i, 4), 16);
/// ```
#[inline]
pub fn pow<T: One + Mul<T, T>>(mut base: T, mut exp: uint) -> T {
    if exp == 1 { base }
    else {
        let mut acc = one::<T>();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = base * base;
            exp = exp >> 1;
        }
        acc
    }
}

/// Numbers which have upper and lower bounds
pub trait Bounded {
    // FIXME (#5527): These should be associated constants
    /// returns the smallest finite number this type can represent
    fn min_value() -> Self;
    /// returns the largest finite number this type can represent
    fn max_value() -> Self;
}

macro_rules! bounded_impl(
    ($t:ty, $min:expr, $max:expr) => {
        impl Bounded for $t {
            #[inline]
            fn min_value() -> $t { $min }

            #[inline]
            fn max_value() -> $t { $max }
        }
    }
)

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

/// Specifies the available operations common to all of Rust's core numeric primitives.
/// These may not always make sense from a purely mathematical point of view, but
/// may be useful for systems programming.
pub trait Primitive: Copy
                   + Clone
                   + Num
                   + NumCast
                   + PartialOrd
                   + Bounded {}

trait_impl!(Primitive for uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

/// A primitive signed or unsigned integer equipped with various bitwise
/// operators, bit counting methods, and endian conversion functions.
pub trait Int: Primitive
             + CheckedAdd
             + CheckedSub
             + CheckedMul
             + CheckedDiv
             + Bounded
             + Not<Self>
             + BitAnd<Self,Self>
             + BitOr<Self,Self>
             + BitXor<Self,Self>
             + Shl<uint,Self>
             + Shr<uint,Self> {
    /// Returns the number of ones in the binary representation of the integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    fn count_ones(self) -> Self;

    /// Returns the number of zeros in the binary representation of the integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_zeros(), 5);
    /// ```
    #[inline]
    fn count_zeros(self) -> Self {
        (!self).count_ones()
    }

    /// Returns the number of leading zeros in the binary representation
    /// of the integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.leading_zeros(), 10);
    /// ```
    fn leading_zeros(self) -> Self;

    /// Returns the number of trailing zeros in the binary representation
    /// of the integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    fn trailing_zeros(self) -> Self;

    /// Shifts the bits to the left by a specified amount amount, `n`, wrapping
    /// the truncated bits to the end of the resulting integer.
    ///
    /// # Example
    ///
    /// ```rust
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
    /// let n = 0x0123456789ABCDEFu64;
    /// let m = 0xEFCDAB8967452301u64;
    ///
    /// assert_eq!(n.swap_bytes(), m);
    /// ```
    fn swap_bytes(self) -> Self;

    /// Convert a integer from big endian to the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
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

    /// Convert a integer from little endian to the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
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

    /// Convert the integer to big endian from the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
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
    #[inline]
    fn to_be(self) -> Self { // or not to be?
        if cfg!(target_endian = "big") { self } else { self.swap_bytes() }
    }

    /// Convert the integer to little endian from the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
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
    #[inline]
    fn to_le(self) -> Self {
        if cfg!(target_endian = "little") { self } else { self.swap_bytes() }
    }
}

macro_rules! int_impl {
    ($T:ty, $BITS:expr, $ctpop:path, $ctlz:path, $cttz:path, $bswap:path) => {
        impl Int for $T {
            #[inline]
            fn count_ones(self) -> $T { unsafe { $ctpop(self) } }

            #[inline]
            fn leading_zeros(self) -> $T { unsafe { $ctlz(self) } }

            #[inline]
            fn trailing_zeros(self) -> $T { unsafe { $cttz(self) } }

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
            fn swap_bytes(self) -> $T { unsafe { $bswap(self) } }
        }
    }
}

/// Swapping a single byte is a no-op. This is marked as `unsafe` for
/// consistency with the other `bswap` intrinsics.
unsafe fn bswap8(x: u8) -> u8 { x }

int_impl!(u8, 8,
    intrinsics::ctpop8,
    intrinsics::ctlz8,
    intrinsics::cttz8,
    bswap8)

int_impl!(u16, 16,
    intrinsics::ctpop16,
    intrinsics::ctlz16,
    intrinsics::cttz16,
    intrinsics::bswap16)

int_impl!(u32, 32,
    intrinsics::ctpop32,
    intrinsics::ctlz32,
    intrinsics::cttz32,
    intrinsics::bswap32)

int_impl!(u64, 64,
    intrinsics::ctpop64,
    intrinsics::ctlz64,
    intrinsics::cttz64,
    intrinsics::bswap64)

macro_rules! int_cast_impl {
    ($T:ty, $U:ty) => {
        impl Int for $T {
            #[inline]
            fn count_ones(self) -> $T { (self as $U).count_ones() as $T }

            #[inline]
            fn leading_zeros(self) -> $T { (self as $U).leading_zeros() as $T }

            #[inline]
            fn trailing_zeros(self) -> $T { (self as $U).trailing_zeros() as $T }

            #[inline]
            fn rotate_left(self, n: uint) -> $T { (self as $U).rotate_left(n) as $T }

            #[inline]
            fn rotate_right(self, n: uint) -> $T { (self as $U).rotate_right(n) as $T }

            #[inline]
            fn swap_bytes(self) -> $T { (self as $U).swap_bytes() as $T }
        }
    }
}

int_cast_impl!(i8, u8)
int_cast_impl!(i16, u16)
int_cast_impl!(i32, u32)
int_cast_impl!(i64, u64)

#[cfg(target_word_size = "32")] int_cast_impl!(uint, u32)
#[cfg(target_word_size = "64")] int_cast_impl!(uint, u64)
#[cfg(target_word_size = "32")] int_cast_impl!(int, u32)
#[cfg(target_word_size = "64")] int_cast_impl!(int, u64)

/// Returns the smallest power of 2 greater than or equal to `n`.
#[inline]
pub fn next_power_of_two<T: Unsigned + Int>(n: T) -> T {
    let halfbits = size_of::<T>() * 4;
    let mut tmp: T = n - one();
    let mut shift = 1u;
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << 1u;
    }
    tmp + one()
}

// Returns `true` iff `n == 2^k` for some k.
#[inline]
pub fn is_power_of_two<T: Unsigned + Int>(n: T) -> bool {
    (n - one()) & n == zero()
}

/// Returns the smallest power of 2 greater than or equal to `n`. If the next
/// power of two is greater than the type's maximum value, `None` is returned,
/// otherwise the power of 2 is wrapped in `Some`.
#[inline]
pub fn checked_next_power_of_two<T: Unsigned + Int>(n: T) -> Option<T> {
    let halfbits = size_of::<T>() * 4;
    let mut tmp: T = n - one();
    let mut shift = 1u;
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << 1u;
    }
    tmp.checked_add(&one())
}

/// A generic trait for converting a value to a number.
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
                let min_value: $DstT = Bounded::min_value();
                let max_value: $DstT = Bounded::max_value();
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
            let zero: $SrcT = Zero::zero();
            let max_value: $DstT = Bounded::max_value();
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
            let max_value: $DstT = Bounded::max_value();
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
                let zero: $SrcT = Zero::zero();
                let max_value: $DstT = Bounded::max_value();
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
            let max_value: $SrcT = Bounded::max_value();
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
pub fn from_int<A: FromPrimitive>(n: int) -> Option<A> {
    FromPrimitive::from_int(n)
}

/// A utility function that just calls `FromPrimitive::from_i8`.
pub fn from_i8<A: FromPrimitive>(n: i8) -> Option<A> {
    FromPrimitive::from_i8(n)
}

/// A utility function that just calls `FromPrimitive::from_i16`.
pub fn from_i16<A: FromPrimitive>(n: i16) -> Option<A> {
    FromPrimitive::from_i16(n)
}

/// A utility function that just calls `FromPrimitive::from_i32`.
pub fn from_i32<A: FromPrimitive>(n: i32) -> Option<A> {
    FromPrimitive::from_i32(n)
}

/// A utility function that just calls `FromPrimitive::from_i64`.
pub fn from_i64<A: FromPrimitive>(n: i64) -> Option<A> {
    FromPrimitive::from_i64(n)
}

/// A utility function that just calls `FromPrimitive::from_uint`.
pub fn from_uint<A: FromPrimitive>(n: uint) -> Option<A> {
    FromPrimitive::from_uint(n)
}

/// A utility function that just calls `FromPrimitive::from_u8`.
pub fn from_u8<A: FromPrimitive>(n: u8) -> Option<A> {
    FromPrimitive::from_u8(n)
}

/// A utility function that just calls `FromPrimitive::from_u16`.
pub fn from_u16<A: FromPrimitive>(n: u16) -> Option<A> {
    FromPrimitive::from_u16(n)
}

/// A utility function that just calls `FromPrimitive::from_u32`.
pub fn from_u32<A: FromPrimitive>(n: u32) -> Option<A> {
    FromPrimitive::from_u32(n)
}

/// A utility function that just calls `FromPrimitive::from_u64`.
pub fn from_u64<A: FromPrimitive>(n: u64) -> Option<A> {
    FromPrimitive::from_u64(n)
}

/// A utility function that just calls `FromPrimitive::from_f32`.
pub fn from_f32<A: FromPrimitive>(n: f32) -> Option<A> {
    FromPrimitive::from_f32(n)
}

/// A utility function that just calls `FromPrimitive::from_f64`.
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
pub fn cast<T: NumCast,U: NumCast>(n: T) -> Option<U> {
    NumCast::from(n)
}

/// An interface for casting between machine scalars.
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

/// Saturating math operations
pub trait Saturating {
    /// Saturating addition operator.
    /// Returns a+b, saturating at the numeric bounds instead of overflowing.
    fn saturating_add(self, v: Self) -> Self;

    /// Saturating subtraction operator.
    /// Returns a-b, saturating at the numeric bounds instead of overflowing.
    fn saturating_sub(self, v: Self) -> Self;
}

impl<T: CheckedAdd + CheckedSub + Zero + PartialOrd + Bounded> Saturating for T {
    #[inline]
    fn saturating_add(self, v: T) -> T {
        match self.checked_add(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::max_value()
            } else {
                Bounded::min_value()
            }
        }
    }

    #[inline]
    fn saturating_sub(self, v: T) -> T {
        match self.checked_sub(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::min_value()
            } else {
                Bounded::max_value()
            }
        }
    }
}

/// Performs addition that returns `None` instead of wrapping around on overflow.
pub trait CheckedAdd: Add<Self, Self> {
    /// Adds two numbers, checking for overflow. If overflow happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::CheckedAdd;
    /// assert_eq!(5u16.checked_add(&65530), Some(65535));
    /// assert_eq!(6u16.checked_add(&65530), None);
    /// ```
    fn checked_add(&self, v: &Self) -> Option<Self>;
}

macro_rules! checked_impl(
    ($trait_name:ident, $method:ident, $t:ty, $op:path) => {
        impl $trait_name for $t {
            #[inline]
            fn $method(&self, v: &$t) -> Option<$t> {
                unsafe {
                    let (x, y) = $op(*self, *v);
                    if y { None } else { Some(x) }
                }
            }
        }
    }
)
macro_rules! checked_cast_impl(
    ($trait_name:ident, $method:ident, $t:ty, $cast:ty, $op:path) => {
        impl $trait_name for $t {
            #[inline]
            fn $method(&self, v: &$t) -> Option<$t> {
                unsafe {
                    let (x, y) = $op(*self as $cast, *v as $cast);
                    if y { None } else { Some(x as $t) }
                }
            }
        }
    }
)

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedAdd, checked_add, uint, u32, intrinsics::u32_add_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedAdd, checked_add, uint, u64, intrinsics::u64_add_with_overflow)

checked_impl!(CheckedAdd, checked_add, u8,  intrinsics::u8_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, u16, intrinsics::u16_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, u32, intrinsics::u32_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, u64, intrinsics::u64_add_with_overflow)

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedAdd, checked_add, int, i32, intrinsics::i32_add_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedAdd, checked_add, int, i64, intrinsics::i64_add_with_overflow)

checked_impl!(CheckedAdd, checked_add, i8,  intrinsics::i8_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, i16, intrinsics::i16_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, i32, intrinsics::i32_add_with_overflow)
checked_impl!(CheckedAdd, checked_add, i64, intrinsics::i64_add_with_overflow)

/// Performs subtraction that returns `None` instead of wrapping around on underflow.
pub trait CheckedSub: Sub<Self, Self> {
    /// Subtracts two numbers, checking for underflow. If underflow happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::CheckedSub;
    /// assert_eq!((-127i8).checked_sub(&1), Some(-128));
    /// assert_eq!((-128i8).checked_sub(&1), None);
    /// ```
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedSub, checked_sub, uint, u32, intrinsics::u32_sub_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedSub, checked_sub, uint, u64, intrinsics::u64_sub_with_overflow)

checked_impl!(CheckedSub, checked_sub, u8,  intrinsics::u8_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, u16, intrinsics::u16_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, u32, intrinsics::u32_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, u64, intrinsics::u64_sub_with_overflow)

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedSub, checked_sub, int, i32, intrinsics::i32_sub_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedSub, checked_sub, int, i64, intrinsics::i64_sub_with_overflow)

checked_impl!(CheckedSub, checked_sub, i8,  intrinsics::i8_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, i16, intrinsics::i16_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, i32, intrinsics::i32_sub_with_overflow)
checked_impl!(CheckedSub, checked_sub, i64, intrinsics::i64_sub_with_overflow)

/// Performs multiplication that returns `None` instead of wrapping around on underflow or
/// overflow.
pub trait CheckedMul: Mul<Self, Self> {
    /// Multiplies two numbers, checking for underflow or overflow. If underflow or overflow
    /// happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::CheckedMul;
    /// assert_eq!(5u8.checked_mul(&51), Some(255));
    /// assert_eq!(5u8.checked_mul(&52), None);
    /// ```
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedMul, checked_mul, uint, u32, intrinsics::u32_mul_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedMul, checked_mul, uint, u64, intrinsics::u64_mul_with_overflow)

checked_impl!(CheckedMul, checked_mul, u8,  intrinsics::u8_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, u16, intrinsics::u16_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, u32, intrinsics::u32_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, u64, intrinsics::u64_mul_with_overflow)

#[cfg(target_word_size = "32")]
checked_cast_impl!(CheckedMul, checked_mul, int, i32, intrinsics::i32_mul_with_overflow)
#[cfg(target_word_size = "64")]
checked_cast_impl!(CheckedMul, checked_mul, int, i64, intrinsics::i64_mul_with_overflow)

checked_impl!(CheckedMul, checked_mul, i8,  intrinsics::i8_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, i16, intrinsics::i16_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, i32, intrinsics::i32_mul_with_overflow)
checked_impl!(CheckedMul, checked_mul, i64, intrinsics::i64_mul_with_overflow)

/// Performs division that returns `None` instead of wrapping around on underflow or overflow.
pub trait CheckedDiv: Div<Self, Self> {
    /// Divides two numbers, checking for underflow or overflow. If underflow or overflow happens,
    /// `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::CheckedDiv;
    /// assert_eq!((-127i8).checked_div(&-1), Some(127));
    /// assert_eq!((-128i8).checked_div(&-1), None);
    /// ```
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

macro_rules! checkeddiv_int_impl(
    ($t:ty, $min:expr) => {
        impl CheckedDiv for $t {
            #[inline]
            fn checked_div(&self, v: &$t) -> Option<$t> {
                if *v == 0 || (*self == $min && *v == -1) {
                    None
                } else {
                    Some(self / *v)
                }
            }
        }
    }
)

checkeddiv_int_impl!(int, int::MIN)
checkeddiv_int_impl!(i8, i8::MIN)
checkeddiv_int_impl!(i16, i16::MIN)
checkeddiv_int_impl!(i32, i32::MIN)
checkeddiv_int_impl!(i64, i64::MIN)

macro_rules! checkeddiv_uint_impl(
    ($($t:ty)*) => ($(
        impl CheckedDiv for $t {
            #[inline]
            fn checked_div(&self, v: &$t) -> Option<$t> {
                if *v == 0 {
                    None
                } else {
                    Some(self / *v)
                }
            }
        }
    )*)
)

checkeddiv_uint_impl!(uint u8 u16 u32 u64)

/// Used for representing the classification of floating point numbers
#[deriving(PartialEq, Show)]
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

/// Operations on primitive floating point numbers.
// FIXME(#5527): In a future version of Rust, many of these functions will
//               become constants.
//
// FIXME(#8888): Several of these functions have a parameter named
//               `unused_self`. Removing it requires #8888 to be fixed.
pub trait Float: Signed + Primitive {
    /// Returns the NaN value.
    fn nan() -> Self;
    /// Returns the infinite value.
    fn infinity() -> Self;
    /// Returns the negative infinite value.
    fn neg_infinity() -> Self;
    /// Returns -0.0.
    fn neg_zero() -> Self;

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
    /// Returns the smallest normalized positive number that this type can represent.
    fn min_pos_value(unused_self: Option<Self>) -> Self;

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
    fn sqrt(self) -> Self;
    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    fn rsqrt(self) -> Self;

    // FIXME (#5527): These should be associated constants

    /// Archimedes' constant.
    fn pi() -> Self;
    /// 2.0 * pi.
    fn two_pi() -> Self;
    /// pi / 2.0.
    fn frac_pi_2() -> Self;
    /// pi / 3.0.
    fn frac_pi_3() -> Self;
    /// pi / 4.0.
    fn frac_pi_4() -> Self;
    /// pi / 6.0.
    fn frac_pi_6() -> Self;
    /// pi / 8.0.
    fn frac_pi_8() -> Self;
    /// 1.0 / pi.
    fn frac_1_pi() -> Self;
    /// 2.0 / pi.
    fn frac_2_pi() -> Self;
    /// 2.0 / sqrt(pi).
    fn frac_2_sqrtpi() -> Self;

    /// Euler's number.
    fn e() -> Self;
    /// log2(e).
    fn log2_e() -> Self;
    /// log10(e).
    fn log10_e() -> Self;
    /// ln(2.0).
    fn ln_2() -> Self;
    /// ln(10.0).
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
