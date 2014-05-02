// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits and functions for generic mathematics
//!
//! These are implemented for the primitive numeric types in `std::{u8, u16,
//! u32, u64, uint, i8, i16, i32, i64, int, f32, f64, float}`.

#![allow(missing_doc)]

use clone::Clone;
use cmp::{Eq, Ord};
use kinds::Copy;
use mem::size_of;
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use ops::{Not, BitAnd, BitOr, BitXor, Shl, Shr};
use option::{Option, Some, None};

/// The base trait for numeric types
pub trait Num: Eq + Zero + One
             + Neg<Self>
             + Add<Self,Self>
             + Sub<Self,Self>
             + Mul<Self,Self>
             + Div<Self,Self>
             + Rem<Self,Self> {}

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
    /// ~~~notrust
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
    fn is_zero(&self) -> bool;
}

/// Returns the additive identity, `0`.
#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

/// Defines a multiplicative identity element for `Self`.
pub trait One: Mul<Self, Self> {
    /// Returns the multiplicative identity element of `Self`, `1`.
    ///
    /// # Laws
    ///
    /// ~~~notrust
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

/// Returns the multiplicative identity, `1`.
#[inline(always)] pub fn one<T: One>() -> T { One::one() }

/// Useful functions for signed numbers (i.e. numbers that can be negative).
pub trait Signed: Num + Neg<Self> {
    /// Computes the absolute value.
    ///
    /// For float, f32, and f64, `NaN` will be returned if the number is `NaN`.
    fn abs(&self) -> Self;

    /// The positive difference of two numbers.
    ///
    /// Returns `zero` if the number is less than or equal to `other`, otherwise the difference
    /// between `self` and `other` is returned.
    fn abs_sub(&self, other: &Self) -> Self;

    /// Returns the sign of the number.
    ///
    /// For `float`, `f32`, `f64`:
    ///   * `1.0` if the number is positive, `+0.0` or `INFINITY`
    ///   * `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    ///   * `NaN` if the number is `NaN`
    ///
    /// For `int`:
    ///   * `0` if the number is zero
    ///   * `1` if the number is positive
    ///   * `-1` if the number is negative
    fn signum(&self) -> Self;

    /// Returns true if the number is positive and false if the number is zero or negative.
    fn is_positive(&self) -> bool;

    /// Returns true if the number is negative and false if the number is zero or positive.
    fn is_negative(&self) -> bool;
}

/// Computes the absolute value.
///
/// For float, f32, and f64, `NaN` will be returned if the number is `NaN`
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
/// For float, f32, f64:
/// - `1.0` if the number is positive, `+0.0` or `INFINITY`
/// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
/// - `NAN` if the number is `NAN`
///
/// For int:
/// - `0` if the number is zero
/// - `1` if the number is positive
/// - `-1` if the number is negative
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

/// A trait for values which cannot be negative
pub trait Unsigned: Num {}

/// Raises a value to the power of exp, using exponentiation by squaring.
///
/// # Example
///
/// ```rust
/// use std::num;
///
/// assert_eq!(num::pow(2, 4), 16);
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

/// Numbers with a fixed binary representation.
pub trait Bitwise: Bounded
                 + Not<Self>
                 + BitAnd<Self,Self>
                 + BitOr<Self,Self>
                 + BitXor<Self,Self>
                 + Shl<Self,Self>
                 + Shr<Self,Self> {
    /// Returns the number of ones in the binary representation of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b01001100u8;
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    fn count_ones(&self) -> Self;

    /// Returns the number of zeros in the binary representation of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b01001100u8;
    /// assert_eq!(n.count_zeros(), 5);
    /// ```
    #[inline]
    fn count_zeros(&self) -> Self {
        (!*self).count_ones()
    }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b0101000u16;
    /// assert_eq!(n.leading_zeros(), 10);
    /// ```
    fn leading_zeros(&self) -> Self;

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::num::Bitwise;
    ///
    /// let n = 0b0101000u16;
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    fn trailing_zeros(&self) -> Self;
}

/// Specifies the available operations common to all of Rust's core numeric primitives.
/// These may not always make sense from a purely mathematical point of view, but
/// may be useful for systems programming.
pub trait Primitive: Copy
                   + Clone
                   + Num
                   + NumCast
                   + Ord
                   + Bounded {}

/// A collection of traits relevant to primitive signed and unsigned integers
pub trait Int: Primitive
             + Bitwise
             + CheckedAdd
             + CheckedSub
             + CheckedMul
             + CheckedDiv {}

/// Returns the smallest power of 2 greater than or equal to `n`.
#[inline]
pub fn next_power_of_two<T: Unsigned + Int>(n: T) -> T {
    let halfbits: T = cast(size_of::<T>() * 4).unwrap();
    let mut tmp: T = n - one();
    let mut shift: T = one();
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << one();
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
    let halfbits: T = cast(size_of::<T>() * 4).unwrap();
    let mut tmp: T = n - one();
    let mut shift: T = one();
    while shift <= halfbits {
        tmp = tmp | (tmp >> shift);
        shift = shift << one();
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
    ($SrcT:ty, $DstT:ty) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some(*self as $DstT)
            } else {
                let n = *self as i64;
                let min_value: $DstT = Bounded::min_value();
                let max_value: $DstT = Bounded::max_value();
                if min_value as i64 <= n && n <= max_value as i64 {
                    Some(*self as $DstT)
                } else {
                    None
                }
            }
        }
    )
)

macro_rules! impl_to_primitive_int_to_uint(
    ($SrcT:ty, $DstT:ty) => (
        {
            let zero: $SrcT = Zero::zero();
            let max_value: $DstT = Bounded::max_value();
            if zero <= *self && *self as u64 <= max_value as u64 {
                Some(*self as $DstT)
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
            fn to_int(&self) -> Option<int> { impl_to_primitive_int_to_int!($T, int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_int_to_int!($T, i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_int_to_int!($T, i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_int_to_int!($T, i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_int_to_int!($T, i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_int_to_uint!($T, uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_int_to_uint!($T, u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_int_to_uint!($T, u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_int_to_uint!($T, u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_int_to_uint!($T, u64) }

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
    ($DstT:ty) => (
        {
            let max_value: $DstT = Bounded::max_value();
            if *self as u64 <= max_value as u64 {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )
)

macro_rules! impl_to_primitive_uint_to_uint(
    ($SrcT:ty, $DstT:ty) => (
        {
            if size_of::<$SrcT>() <= size_of::<$DstT>() {
                Some(*self as $DstT)
            } else {
                let zero: $SrcT = Zero::zero();
                let max_value: $DstT = Bounded::max_value();
                if zero <= *self && *self as u64 <= max_value as u64 {
                    Some(*self as $DstT)
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
            fn to_int(&self) -> Option<int> { impl_to_primitive_uint_to_int!(int) }
            #[inline]
            fn to_i8(&self) -> Option<i8> { impl_to_primitive_uint_to_int!(i8) }
            #[inline]
            fn to_i16(&self) -> Option<i16> { impl_to_primitive_uint_to_int!(i16) }
            #[inline]
            fn to_i32(&self) -> Option<i32> { impl_to_primitive_uint_to_int!(i32) }
            #[inline]
            fn to_i64(&self) -> Option<i64> { impl_to_primitive_uint_to_int!(i64) }

            #[inline]
            fn to_uint(&self) -> Option<uint> { impl_to_primitive_uint_to_uint!($T, uint) }
            #[inline]
            fn to_u8(&self) -> Option<u8> { impl_to_primitive_uint_to_uint!($T, u8) }
            #[inline]
            fn to_u16(&self) -> Option<u16> { impl_to_primitive_uint_to_uint!($T, u16) }
            #[inline]
            fn to_u32(&self) -> Option<u32> { impl_to_primitive_uint_to_uint!($T, u32) }
            #[inline]
            fn to_u64(&self) -> Option<u64> { impl_to_primitive_uint_to_uint!($T, u64) }

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
    ($SrcT:ty, $DstT:ty) => (
        if size_of::<$SrcT>() <= size_of::<$DstT>() {
            Some(*self as $DstT)
        } else {
            let n = *self as f64;
            let max_value: $SrcT = Bounded::max_value();
            if -max_value as f64 <= n && n <= max_value as f64 {
                Some(*self as $DstT)
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
            fn to_f32(&self) -> Option<f32> { impl_to_primitive_float_to_float!($T, f32) }
            #[inline]
            fn to_f64(&self) -> Option<f64> { impl_to_primitive_float_to_float!($T, f64) }
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
    ($T:ty, $to_ty:expr) => (
        impl FromPrimitive for $T {
            #[inline] fn from_int(n: int) -> Option<$T> { $to_ty }
            #[inline] fn from_i8(n: i8) -> Option<$T> { $to_ty }
            #[inline] fn from_i16(n: i16) -> Option<$T> { $to_ty }
            #[inline] fn from_i32(n: i32) -> Option<$T> { $to_ty }
            #[inline] fn from_i64(n: i64) -> Option<$T> { $to_ty }

            #[inline] fn from_uint(n: uint) -> Option<$T> { $to_ty }
            #[inline] fn from_u8(n: u8) -> Option<$T> { $to_ty }
            #[inline] fn from_u16(n: u16) -> Option<$T> { $to_ty }
            #[inline] fn from_u32(n: u32) -> Option<$T> { $to_ty }
            #[inline] fn from_u64(n: u64) -> Option<$T> { $to_ty }

            #[inline] fn from_f32(n: f32) -> Option<$T> { $to_ty }
            #[inline] fn from_f64(n: f64) -> Option<$T> { $to_ty }
        }
    )
)

impl_from_primitive!(int, n.to_int())
impl_from_primitive!(i8, n.to_i8())
impl_from_primitive!(i16, n.to_i16())
impl_from_primitive!(i32, n.to_i32())
impl_from_primitive!(i64, n.to_i64())
impl_from_primitive!(uint, n.to_uint())
impl_from_primitive!(u8, n.to_u8())
impl_from_primitive!(u16, n.to_u16())
impl_from_primitive!(u32, n.to_u32())
impl_from_primitive!(u64, n.to_u64())
impl_from_primitive!(f32, n.to_f32())
impl_from_primitive!(f64, n.to_f64())

/// Cast from one machine scalar to another.
///
/// # Example
///
/// ```
/// use std::num;
///
/// let twenty: f32 = num::cast(0x14).unwrap();
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

impl<T: CheckedAdd + CheckedSub + Zero + Ord + Bounded> Saturating for T {
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
    fn checked_add(&self, v: &Self) -> Option<Self>;
}

/// Performs subtraction that returns `None` instead of wrapping around on underflow.
pub trait CheckedSub: Sub<Self, Self> {
    /// Subtracts two numbers, checking for underflow. If underflow happens, `None` is returned.
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

/// Performs multiplication that returns `None` instead of wrapping around on underflow or
/// overflow.
pub trait CheckedMul: Mul<Self, Self> {
    /// Multiplies two numbers, checking for underflow or overflow. If underflow or overflow
    /// happens, `None` is returned.
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

/// Performs division that returns `None` instead of wrapping around on underflow or overflow.
pub trait CheckedDiv: Div<Self, Self> {
    /// Divides two numbers, checking for underflow or overflow. If underflow or overflow happens,
    /// `None` is returned.
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T:Num + NumCast + ::std::fmt::Show>(ten: T, two: T) {
    assert_eq!(ten.add(&two),  cast(12).unwrap());
    assert_eq!(ten.sub(&two),  cast(8).unwrap());
    assert_eq!(ten.mul(&two),  cast(20).unwrap());
    assert_eq!(ten.div(&two),  cast(5).unwrap());
    assert_eq!(ten.rem(&two),  cast(0).unwrap());

    assert_eq!(ten.add(&two),  ten + two);
    assert_eq!(ten.sub(&two),  ten - two);
    assert_eq!(ten.mul(&two),  ten * two);
    assert_eq!(ten.div(&two),  ten / two);
    assert_eq!(ten.rem(&two),  ten % two);
}
