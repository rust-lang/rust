// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Overloadable operators
//!
//! Implementing these traits allows you to get an effect similar to
//! overloading operators.
//!
//! Some of these traits are imported by the prelude, so they are available in
//! every Rust program.
//!
//! Many of the operators take their operands by value. In non-generic
//! contexts involving built-in types, this is usually not a problem.
//! However, using these operators in generic code, requires some
//! attention if values have to be reused as opposed to letting the operators
//! consume them. One option is to occasionally use `clone()`.
//! Another option is to rely on the types involved providing additional
//! operator implementations for references. For example, for a user-defined
//! type `T` which is supposed to support addition, it is probably a good
//! idea to have both `T` and `&T` implement the traits `Add<T>` and `Add<&T>`
//! so that generic code can be written without unnecessary cloning.
//!
//! # Examples
//!
//! This example creates a `Point` struct that implements `Add` and `Sub`, and
//! then demonstrates adding and subtracting two `Point`s.
//!
//! ```rust
//! use std::ops::{Add, Sub};
//!
//! #[derive(Debug)]
//! struct Point {
//!     x: i32,
//!     y: i32
//! }
//!
//! impl Add for Point {
//!     type Output = Point;
//!
//!     fn add(self, other: Point) -> Point {
//!         Point {x: self.x + other.x, y: self.y + other.y}
//!     }
//! }
//!
//! impl Sub for Point {
//!     type Output = Point;
//!
//!     fn sub(self, other: Point) -> Point {
//!         Point {x: self.x - other.x, y: self.y - other.y}
//!     }
//! }
//! fn main() {
//!     println!("{:?}", Point {x: 1, y: 0} + Point {x: 2, y: 3});
//!     println!("{:?}", Point {x: 1, y: 0} - Point {x: 2, y: 3});
//! }
//! ```
//!
//! See the documentation for each trait for a minimum implementation that
//! prints something to the screen.

#![stable(feature = "rust1", since = "1.0.0")]

use marker::{Sized, Unsize};
use fmt;

/// The `Drop` trait is used to run some code when a value goes out of scope.
/// This is sometimes called a 'destructor'.
///
/// # Examples
///
/// A trivial implementation of `Drop`. The `drop` method is called when `_x`
/// goes out of scope, and therefore `main` prints `Dropping!`.
///
/// ```
/// struct HasDrop;
///
/// impl Drop for HasDrop {
///     fn drop(&mut self) {
///         println!("Dropping!");
///     }
/// }
///
/// fn main() {
///     let _x = HasDrop;
/// }
/// ```
#[lang = "drop"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Drop {
    /// A method called when the value goes out of scope.
    ///
    /// When this method has been called, `self` has not yet been deallocated.
    /// If it were, `self` would be a dangling reference.
    ///
    /// After this function is over, the memory of `self` will be deallocated.
    ///
    /// # Panics
    ///
    /// Given that a `panic!` will call `drop()` as it unwinds, any `panic!` in
    /// a `drop()` implementation will likely abort.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}

// implements the unary operator "op &T"
// based on "op T" where T is expected to be `Copy`able
macro_rules! forward_ref_unop {
    (impl $imp:ident, $method:ident for $t:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> $imp for &'a $t {
            type Output = <$t as $imp>::Output;

            #[inline]
            fn $method(self) -> <$t as $imp>::Output {
                $imp::$method(*self)
            }
        }
    }
}

// implements binary operators "&T op U", "T op &U", "&T op &U"
// based on "T op U" where T and U are expected to be `Copy`able
macro_rules! forward_ref_binop {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> $imp<$u> for &'a $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, other)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> $imp<&'a $u> for $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &'a $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self, *other)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b> $imp<&'a $u> for &'b $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &'a $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, *other)
            }
        }
    }
}

/// The `Add` trait is used to specify the functionality of `+`.
///
/// # Examples
///
/// A trivial implementation of `Add`. When `Foo + Foo` happens, it ends up
/// calling `add`, and therefore, `main` prints `Adding!`.
///
/// ```
/// use std::ops::Add;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Add for Foo {
///     type Output = Foo;
///
///     fn add(self, _rhs: Foo) -> Foo {
///         println!("Adding!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo + Foo;
/// }
/// ```
#[lang = "add"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Add<RHS=Self> {
    /// The resulting type after applying the `+` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `+` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn add(self, rhs: RHS) -> Self::Output;
}

macro_rules! add_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Add for $t {
            type Output = $t;

            #[inline]
            fn add(self, other: $t) -> $t { self + other }
        }

        forward_ref_binop! { impl Add, add for $t, $t }
    )*)
}

add_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `Sub` trait is used to specify the functionality of `-`.
///
/// # Examples
///
/// A trivial implementation of `Sub`. When `Foo - Foo` happens, it ends up
/// calling `sub`, and therefore, `main` prints `Subtracting!`.
///
/// ```
/// use std::ops::Sub;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Sub for Foo {
///     type Output = Foo;
///
///     fn sub(self, _rhs: Foo) -> Foo {
///         println!("Subtracting!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo - Foo;
/// }
/// ```
#[lang = "sub"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Sub<RHS=Self> {
    /// The resulting type after applying the `-` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `-` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sub(self, rhs: RHS) -> Self::Output;
}

macro_rules! sub_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Sub for $t {
            type Output = $t;

            #[inline]
            fn sub(self, other: $t) -> $t { self - other }
        }

        forward_ref_binop! { impl Sub, sub for $t, $t }
    )*)
}

sub_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `Mul` trait is used to specify the functionality of `*`.
///
/// # Examples
///
/// A trivial implementation of `Mul`. When `Foo * Foo` happens, it ends up
/// calling `mul`, and therefore, `main` prints `Multiplying!`.
///
/// ```
/// use std::ops::Mul;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Mul for Foo {
///     type Output = Foo;
///
///     fn mul(self, _rhs: Foo) -> Foo {
///         println!("Multiplying!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo * Foo;
/// }
/// ```
#[lang = "mul"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Mul<RHS=Self> {
    /// The resulting type after applying the `*` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `*` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn mul(self, rhs: RHS) -> Self::Output;
}

macro_rules! mul_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Mul for $t {
            type Output = $t;

            #[inline]
            fn mul(self, other: $t) -> $t { self * other }
        }

        forward_ref_binop! { impl Mul, mul for $t, $t }
    )*)
}

mul_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `Div` trait is used to specify the functionality of `/`.
///
/// # Examples
///
/// A trivial implementation of `Div`. When `Foo / Foo` happens, it ends up
/// calling `div`, and therefore, `main` prints `Dividing!`.
///
/// ```
/// use std::ops::Div;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Div for Foo {
///     type Output = Foo;
///
///     fn div(self, _rhs: Foo) -> Foo {
///         println!("Dividing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo / Foo;
/// }
/// ```
#[lang = "div"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Div<RHS=Self> {
    /// The resulting type after applying the `/` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `/` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn div(self, rhs: RHS) -> Self::Output;
}

macro_rules! div_impl_integer {
    ($($t:ty)*) => ($(
        /// This operation rounds towards zero, truncating any
        /// fractional part of the exact result.
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Div for $t {
            type Output = $t;

            #[inline]
            fn div(self, other: $t) -> $t { self / other }
        }

        forward_ref_binop! { impl Div, div for $t, $t }
    )*)
}

div_impl_integer! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

macro_rules! div_impl_float {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Div for $t {
            type Output = $t;

            #[inline]
            fn div(self, other: $t) -> $t { self / other }
        }

        forward_ref_binop! { impl Div, div for $t, $t }
    )*)
}

div_impl_float! { f32 f64 }

/// The `Rem` trait is used to specify the functionality of `%`.
///
/// # Examples
///
/// A trivial implementation of `Rem`. When `Foo % Foo` happens, it ends up
/// calling `rem`, and therefore, `main` prints `Remainder-ing!`.
///
/// ```
/// use std::ops::Rem;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Rem for Foo {
///     type Output = Foo;
///
///     fn rem(self, _rhs: Foo) -> Foo {
///         println!("Remainder-ing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo % Foo;
/// }
/// ```
#[lang = "rem"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Rem<RHS=Self> {
    /// The resulting type after applying the `%` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output = Self;

    /// The method for the `%` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rem(self, rhs: RHS) -> Self::Output;
}

macro_rules! rem_impl_integer {
    ($($t:ty)*) => ($(
        /// This operation satisfies `n % d == n - (n / d) * d`.  The
        /// result has the same sign as the left operand.
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Rem for $t {
            type Output = $t;

            #[inline]
            fn rem(self, other: $t) -> $t { self % other }
        }

        forward_ref_binop! { impl Rem, rem for $t, $t }
    )*)
}

rem_impl_integer! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

macro_rules! rem_impl_float {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Rem for $t {
            type Output = $t;

            #[inline]
            fn rem(self, other: $t) -> $t { self % other }
        }

        forward_ref_binop! { impl Rem, rem for $t, $t }
    )*)
}

rem_impl_float! { f32 f64 }

/// The `Neg` trait is used to specify the functionality of unary `-`.
///
/// # Examples
///
/// A trivial implementation of `Neg`. When `-Foo` happens, it ends up calling
/// `neg`, and therefore, `main` prints `Negating!`.
///
/// ```
/// use std::ops::Neg;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Neg for Foo {
///     type Output = Foo;
///
///     fn neg(self) -> Foo {
///         println!("Negating!");
///         self
///     }
/// }
///
/// fn main() {
///     -Foo;
/// }
/// ```
#[lang = "neg"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Neg {
    /// The resulting type after applying the `-` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the unary `-` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn neg(self) -> Self::Output;
}



macro_rules! neg_impl_core {
    ($id:ident => $body:expr, $($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Neg for $t {
            type Output = $t;

            #[inline]
            fn neg(self) -> $t { let $id = self; $body }
        }

        forward_ref_unop! { impl Neg, neg for $t }
    )*)
}

macro_rules! neg_impl_numeric {
    ($($t:ty)*) => { neg_impl_core!{ x => -x, $($t)*} }
}

macro_rules! neg_impl_unsigned {
    ($($t:ty)*) => {
        neg_impl_core!{ x => {
            !x.wrapping_add(1)
        }, $($t)*} }
}

// neg_impl_unsigned! { usize u8 u16 u32 u64 }
neg_impl_numeric! { isize i8 i16 i32 i64 f32 f64 }

/// The `Not` trait is used to specify the functionality of unary `!`.
///
/// # Examples
///
/// A trivial implementation of `Not`. When `!Foo` happens, it ends up calling
/// `not`, and therefore, `main` prints `Not-ing!`.
///
/// ```
/// use std::ops::Not;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Not for Foo {
///     type Output = Foo;
///
///     fn not(self) -> Foo {
///         println!("Not-ing!");
///         self
///     }
/// }
///
/// fn main() {
///     !Foo;
/// }
/// ```
#[lang = "not"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Not {
    /// The resulting type after applying the `!` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the unary `!` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn not(self) -> Self::Output;
}

macro_rules! not_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Not for $t {
            type Output = $t;

            #[inline]
            fn not(self) -> $t { !self }
        }

        forward_ref_unop! { impl Not, not for $t }
    )*)
}

not_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `BitAnd` trait is used to specify the functionality of `&`.
///
/// # Examples
///
/// A trivial implementation of `BitAnd`. When `Foo & Foo` happens, it ends up
/// calling `bitand`, and therefore, `main` prints `Bitwise And-ing!`.
///
/// ```
/// use std::ops::BitAnd;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitAnd for Foo {
///     type Output = Foo;
///
///     fn bitand(self, _rhs: Foo) -> Foo {
///         println!("Bitwise And-ing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo & Foo;
/// }
/// ```
#[lang = "bitand"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait BitAnd<RHS=Self> {
    /// The resulting type after applying the `&` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `&` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitand(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitand_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitAnd for $t {
            type Output = $t;

            #[inline]
            fn bitand(self, rhs: $t) -> $t { self & rhs }
        }

        forward_ref_binop! { impl BitAnd, bitand for $t, $t }
    )*)
}

bitand_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `BitOr` trait is used to specify the functionality of `|`.
///
/// # Examples
///
/// A trivial implementation of `BitOr`. When `Foo | Foo` happens, it ends up
/// calling `bitor`, and therefore, `main` prints `Bitwise Or-ing!`.
///
/// ```
/// use std::ops::BitOr;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitOr for Foo {
///     type Output = Foo;
///
///     fn bitor(self, _rhs: Foo) -> Foo {
///         println!("Bitwise Or-ing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo | Foo;
/// }
/// ```
#[lang = "bitor"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait BitOr<RHS=Self> {
    /// The resulting type after applying the `|` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `|` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitor(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitor_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for $t {
            type Output = $t;

            #[inline]
            fn bitor(self, rhs: $t) -> $t { self | rhs }
        }

        forward_ref_binop! { impl BitOr, bitor for $t, $t }
    )*)
}

bitor_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `BitXor` trait is used to specify the functionality of `^`.
///
/// # Examples
///
/// A trivial implementation of `BitXor`. When `Foo ^ Foo` happens, it ends up
/// calling `bitxor`, and therefore, `main` prints `Bitwise Xor-ing!`.
///
/// ```
/// use std::ops::BitXor;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitXor for Foo {
///     type Output = Foo;
///
///     fn bitxor(self, _rhs: Foo) -> Foo {
///         println!("Bitwise Xor-ing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo ^ Foo;
/// }
/// ```
#[lang = "bitxor"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait BitXor<RHS=Self> {
    /// The resulting type after applying the `^` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `^` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitxor(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitxor_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitXor for $t {
            type Output = $t;

            #[inline]
            fn bitxor(self, other: $t) -> $t { self ^ other }
        }

        forward_ref_binop! { impl BitXor, bitxor for $t, $t }
    )*)
}

bitxor_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `Shl` trait is used to specify the functionality of `<<`.
///
/// # Examples
///
/// A trivial implementation of `Shl`. When `Foo << Foo` happens, it ends up
/// calling `shl`, and therefore, `main` prints `Shifting left!`.
///
/// ```
/// use std::ops::Shl;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Shl<Foo> for Foo {
///     type Output = Foo;
///
///     fn shl(self, _rhs: Foo) -> Foo {
///         println!("Shifting left!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo << Foo;
/// }
/// ```
#[lang = "shl"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Shl<RHS> {
    /// The resulting type after applying the `<<` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `<<` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn shl(self, rhs: RHS) -> Self::Output;
}

macro_rules! shl_impl {
    ($t:ty, $f:ty) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for $t {
            type Output = $t;

            #[inline]
            fn shl(self, other: $f) -> $t {
                self << other
            }
        }

        forward_ref_binop! { impl Shl, shl for $t, $f }
    )
}

macro_rules! shl_impl_all {
    ($($t:ty)*) => ($(
        shl_impl! { $t, u8 }
        shl_impl! { $t, u16 }
        shl_impl! { $t, u32 }
        shl_impl! { $t, u64 }
        shl_impl! { $t, usize }

        shl_impl! { $t, i8 }
        shl_impl! { $t, i16 }
        shl_impl! { $t, i32 }
        shl_impl! { $t, i64 }
        shl_impl! { $t, isize }
    )*)
}

shl_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

/// The `Shr` trait is used to specify the functionality of `>>`.
///
/// # Examples
///
/// A trivial implementation of `Shr`. When `Foo >> Foo` happens, it ends up
/// calling `shr`, and therefore, `main` prints `Shifting right!`.
///
/// ```
/// use std::ops::Shr;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl Shr<Foo> for Foo {
///     type Output = Foo;
///
///     fn shr(self, _rhs: Foo) -> Foo {
///         println!("Shifting right!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo >> Foo;
/// }
/// ```
#[lang = "shr"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Shr<RHS> {
    /// The resulting type after applying the `>>` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// The method for the `>>` operator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn shr(self, rhs: RHS) -> Self::Output;
}

macro_rules! shr_impl {
    ($t:ty, $f:ty) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for $t {
            type Output = $t;

            #[inline]
            fn shr(self, other: $f) -> $t {
                self >> other
            }
        }

        forward_ref_binop! { impl Shr, shr for $t, $f }
    )
}

macro_rules! shr_impl_all {
    ($($t:ty)*) => ($(
        shr_impl! { $t, u8 }
        shr_impl! { $t, u16 }
        shr_impl! { $t, u32 }
        shr_impl! { $t, u64 }
        shr_impl! { $t, usize }

        shr_impl! { $t, i8 }
        shr_impl! { $t, i16 }
        shr_impl! { $t, i32 }
        shr_impl! { $t, i64 }
        shr_impl! { $t, isize }
    )*)
}

shr_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

/// The `AddAssign` trait is used to specify the functionality of `+=`.
///
/// # Examples
///
/// A trivial implementation of `AddAssign`. When `Foo += Foo` happens, it ends up
/// calling `add_assign`, and therefore, `main` prints `Adding!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::AddAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl AddAssign for Foo {
///     fn add_assign(&mut self, _rhs: Foo) {
///         println!("Adding!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo += Foo;
/// }
/// ```
#[lang = "add_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait AddAssign<Rhs=Self> {
    /// The method for the `+=` operator
    fn add_assign(&mut self, Rhs);
}

macro_rules! add_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl AddAssign for $t {
            #[inline]
            fn add_assign(&mut self, other: $t) { *self += other }
        }
    )+)
}

add_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `SubAssign` trait is used to specify the functionality of `-=`.
///
/// # Examples
///
/// A trivial implementation of `SubAssign`. When `Foo -= Foo` happens, it ends up
/// calling `sub_assign`, and therefore, `main` prints `Subtracting!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::SubAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl SubAssign for Foo {
///     fn sub_assign(&mut self, _rhs: Foo) {
///         println!("Subtracting!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo -= Foo;
/// }
/// ```
#[lang = "sub_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait SubAssign<Rhs=Self> {
    /// The method for the `-=` operator
    fn sub_assign(&mut self, Rhs);
}

macro_rules! sub_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl SubAssign for $t {
            #[inline]
            fn sub_assign(&mut self, other: $t) { *self -= other }
        }
    )+)
}

sub_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `MulAssign` trait is used to specify the functionality of `*=`.
///
/// # Examples
///
/// A trivial implementation of `MulAssign`. When `Foo *= Foo` happens, it ends up
/// calling `mul_assign`, and therefore, `main` prints `Multiplying!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::MulAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl MulAssign for Foo {
///     fn mul_assign(&mut self, _rhs: Foo) {
///         println!("Multiplying!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo *= Foo;
/// }
/// ```
#[lang = "mul_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait MulAssign<Rhs=Self> {
    /// The method for the `*=` operator
    fn mul_assign(&mut self, Rhs);
}

macro_rules! mul_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl MulAssign for $t {
            #[inline]
            fn mul_assign(&mut self, other: $t) { *self *= other }
        }
    )+)
}

mul_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `DivAssign` trait is used to specify the functionality of `/=`.
///
/// # Examples
///
/// A trivial implementation of `DivAssign`. When `Foo /= Foo` happens, it ends up
/// calling `div_assign`, and therefore, `main` prints `Dividing!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::DivAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl DivAssign for Foo {
///     fn div_assign(&mut self, _rhs: Foo) {
///         println!("Dividing!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo /= Foo;
/// }
/// ```
#[lang = "div_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait DivAssign<Rhs=Self> {
    /// The method for the `/=` operator
    fn div_assign(&mut self, Rhs);
}

macro_rules! div_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl DivAssign for $t {
            #[inline]
            fn div_assign(&mut self, other: $t) { *self /= other }
        }
    )+)
}

div_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `RemAssign` trait is used to specify the functionality of `%=`.
///
/// # Examples
///
/// A trivial implementation of `RemAssign`. When `Foo %= Foo` happens, it ends up
/// calling `rem_assign`, and therefore, `main` prints `Remainder-ing!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::RemAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl RemAssign for Foo {
///     fn rem_assign(&mut self, _rhs: Foo) {
///         println!("Remainder-ing!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo %= Foo;
/// }
/// ```
#[lang = "rem_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait RemAssign<Rhs=Self> {
    /// The method for the `%=` operator
    fn rem_assign(&mut self, Rhs);
}

macro_rules! rem_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl RemAssign for $t {
            #[inline]
            fn rem_assign(&mut self, other: $t) { *self %= other }
        }
    )+)
}

rem_assign_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

/// The `BitAndAssign` trait is used to specify the functionality of `&=`.
///
/// # Examples
///
/// A trivial implementation of `BitAndAssign`. When `Foo &= Foo` happens, it ends up
/// calling `bitand_assign`, and therefore, `main` prints `Bitwise And-ing!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::BitAndAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitAndAssign for Foo {
///     fn bitand_assign(&mut self, _rhs: Foo) {
///         println!("Bitwise And-ing!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo &= Foo;
/// }
/// ```
#[lang = "bitand_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait BitAndAssign<Rhs=Self> {
    /// The method for the `&` operator
    fn bitand_assign(&mut self, Rhs);
}

macro_rules! bitand_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitAndAssign for $t {
            #[inline]
            fn bitand_assign(&mut self, other: $t) { *self &= other }
        }
    )+)
}

bitand_assign_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `BitOrAssign` trait is used to specify the functionality of `|=`.
///
/// # Examples
///
/// A trivial implementation of `BitOrAssign`. When `Foo |= Foo` happens, it ends up
/// calling `bitor_assign`, and therefore, `main` prints `Bitwise Or-ing!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::BitOrAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitOrAssign for Foo {
///     fn bitor_assign(&mut self, _rhs: Foo) {
///         println!("Bitwise Or-ing!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo |= Foo;
/// }
/// ```
#[lang = "bitor_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait BitOrAssign<Rhs=Self> {
    /// The method for the `|=` operator
    fn bitor_assign(&mut self, Rhs);
}

macro_rules! bitor_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitOrAssign for $t {
            #[inline]
            fn bitor_assign(&mut self, other: $t) { *self |= other }
        }
    )+)
}

bitor_assign_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `BitXorAssign` trait is used to specify the functionality of `^=`.
///
/// # Examples
///
/// A trivial implementation of `BitXorAssign`. When `Foo ^= Foo` happens, it ends up
/// calling `bitxor_assign`, and therefore, `main` prints `Bitwise Xor-ing!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::BitXorAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl BitXorAssign for Foo {
///     fn bitxor_assign(&mut self, _rhs: Foo) {
///         println!("Bitwise Xor-ing!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo ^= Foo;
/// }
/// ```
#[lang = "bitxor_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait BitXorAssign<Rhs=Self> {
    /// The method for the `^=` operator
    fn bitxor_assign(&mut self, Rhs);
}

macro_rules! bitxor_assign_impl {
    ($($t:ty)+) => ($(
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitXorAssign for $t {
            #[inline]
            fn bitxor_assign(&mut self, other: $t) { *self ^= other }
        }
    )+)
}

bitxor_assign_impl! { bool usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

/// The `ShlAssign` trait is used to specify the functionality of `<<=`.
///
/// # Examples
///
/// A trivial implementation of `ShlAssign`. When `Foo <<= Foo` happens, it ends up
/// calling `shl_assign`, and therefore, `main` prints `Shifting left!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::ShlAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl ShlAssign<Foo> for Foo {
///     fn shl_assign(&mut self, _rhs: Foo) {
///         println!("Shifting left!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo <<= Foo;
/// }
/// ```
#[lang = "shl_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait ShlAssign<Rhs> {
    /// The method for the `<<=` operator
    fn shl_assign(&mut self, Rhs);
}

macro_rules! shl_assign_impl {
    ($t:ty, $f:ty) => (
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl ShlAssign<$f> for $t {
            #[inline]
            fn shl_assign(&mut self, other: $f) {
                *self <<= other
            }
        }
    )
}

macro_rules! shl_assign_impl_all {
    ($($t:ty)*) => ($(
        shl_assign_impl! { $t, u8 }
        shl_assign_impl! { $t, u16 }
        shl_assign_impl! { $t, u32 }
        shl_assign_impl! { $t, u64 }
        shl_assign_impl! { $t, usize }

        shl_assign_impl! { $t, i8 }
        shl_assign_impl! { $t, i16 }
        shl_assign_impl! { $t, i32 }
        shl_assign_impl! { $t, i64 }
        shl_assign_impl! { $t, isize }
    )*)
}

shl_assign_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

/// The `ShrAssign` trait is used to specify the functionality of `>>=`.
///
/// # Examples
///
/// A trivial implementation of `ShrAssign`. When `Foo >>= Foo` happens, it ends up
/// calling `shr_assign`, and therefore, `main` prints `Shifting right!`.
///
/// ```
/// #![feature(augmented_assignments)]
/// #![feature(op_assign_traits)]
///
/// use std::ops::ShrAssign;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
///
/// impl ShrAssign<Foo> for Foo {
///     fn shr_assign(&mut self, _rhs: Foo) {
///         println!("Shifting right!");
///     }
/// }
///
/// # #[allow(unused_assignments)]
/// fn main() {
///     let mut foo = Foo;
///     foo >>= Foo;
/// }
/// ```
#[lang = "shr_assign"]
#[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
pub trait ShrAssign<Rhs=Self> {
    /// The method for the `>>=` operator
    fn shr_assign(&mut self, Rhs);
}

macro_rules! shr_assign_impl {
    ($t:ty, $f:ty) => (
        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl ShrAssign<$f> for $t {
            #[inline]
            fn shr_assign(&mut self, other: $f) {
                *self >>= other
            }
        }
    )
}

macro_rules! shr_assign_impl_all {
    ($($t:ty)*) => ($(
        shr_assign_impl! { $t, u8 }
        shr_assign_impl! { $t, u16 }
        shr_assign_impl! { $t, u32 }
        shr_assign_impl! { $t, u64 }
        shr_assign_impl! { $t, usize }

        shr_assign_impl! { $t, i8 }
        shr_assign_impl! { $t, i16 }
        shr_assign_impl! { $t, i32 }
        shr_assign_impl! { $t, i64 }
        shr_assign_impl! { $t, isize }
    )*)
}

shr_assign_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

/// The `Index` trait is used to specify the functionality of indexing operations
/// like `arr[idx]` when used in an immutable context.
///
/// # Examples
///
/// A trivial implementation of `Index`. When `Foo[Bar]` happens, it ends up
/// calling `index`, and therefore, `main` prints `Indexing!`.
///
/// ```
/// use std::ops::Index;
///
/// #[derive(Copy, Clone)]
/// struct Foo;
/// struct Bar;
///
/// impl Index<Bar> for Foo {
///     type Output = Foo;
///
///     fn index<'a>(&'a self, _index: Bar) -> &'a Foo {
///         println!("Indexing!");
///         self
///     }
/// }
///
/// fn main() {
///     Foo[Bar];
/// }
/// ```
#[lang = "index"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Index<Idx: ?Sized> {
    /// The returned type after indexing
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output: ?Sized;

    /// The method for the indexing (`Foo[Bar]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index(&self, index: Idx) -> &Self::Output;
}

/// The `IndexMut` trait is used to specify the functionality of indexing
/// operations like `arr[idx]`, when used in a mutable context.
///
/// # Examples
///
/// A trivial implementation of `IndexMut`. When `Foo[Bar]` happens, it ends up
/// calling `index_mut`, and therefore, `main` prints `Indexing!`.
///
/// ```
/// use std::ops::{Index, IndexMut};
///
/// #[derive(Copy, Clone)]
/// struct Foo;
/// struct Bar;
///
/// impl Index<Bar> for Foo {
///     type Output = Foo;
///
///     fn index<'a>(&'a self, _index: Bar) -> &'a Foo {
///         self
///     }
/// }
///
/// impl IndexMut<Bar> for Foo {
///     fn index_mut<'a>(&'a mut self, _index: Bar) -> &'a mut Foo {
///         println!("Indexing!");
///         self
///     }
/// }
///
/// fn main() {
///     &mut Foo[Bar];
/// }
/// ```
#[lang = "index_mut"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be mutably indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    /// The method for the indexing (`Foo[Bar]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}

/// An unbounded range.
#[derive(Copy, Clone, PartialEq, Eq)]
#[lang = "range_full"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFull;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for RangeFull {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..")
    }
}

/// A (half-open) range which is bounded at both ends.
#[derive(Clone, PartialEq, Eq)]
#[lang = "range"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Range<Idx> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub start: Idx,
    /// The upper bound of the range (exclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub end: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for Range<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..{:?}", self.start, self.end)
    }
}

/// A range which is only bounded below.
#[derive(Clone, PartialEq, Eq)]
#[lang = "range_from"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFrom<Idx> {
    /// The lower bound of the range (inclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub start: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeFrom<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}..", self.start)
    }
}

/// A range which is only bounded above.
#[derive(Copy, Clone, PartialEq, Eq)]
#[lang = "range_to"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeTo<Idx> {
    /// The upper bound of the range (exclusive).
    #[stable(feature = "rust1", since = "1.0.0")]
    pub end: Idx,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<Idx: fmt::Debug> fmt::Debug for RangeTo<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..{:?}", self.end)
    }
}

/// The `Deref` trait is used to specify the functionality of dereferencing
/// operations, like `*v`.
///
/// `Deref` also enables ['`Deref` coercions'][coercions].
///
/// [coercions]: ../../book/deref-coercions.html
///
/// # Examples
///
/// A struct with a single field which is accessible via dereferencing the
/// struct.
///
/// ```
/// use std::ops::Deref;
///
/// struct DerefExample<T> {
///     value: T
/// }
///
/// impl<T> Deref for DerefExample<T> {
///     type Target = T;
///
///     fn deref(&self) -> &T {
///         &self.value
///     }
/// }
///
/// fn main() {
///     let x = DerefExample { value: 'a' };
///     assert_eq!('a', *x);
/// }
/// ```
#[lang = "deref"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Deref {
    /// The resulting type after dereferencing
    #[stable(feature = "rust1", since = "1.0.0")]
    type Target: ?Sized;

    /// The method called to dereference a value
    #[stable(feature = "rust1", since = "1.0.0")]
    fn deref(&self) -> &Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Deref for &'a T {
    type Target = T;

    fn deref(&self) -> &T { *self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Deref for &'a mut T {
    type Target = T;

    fn deref(&self) -> &T { *self }
}

/// The `DerefMut` trait is used to specify the functionality of dereferencing
/// mutably like `*v = 1;`
///
/// `DerefMut` also enables ['`Deref` coercions'][coercions].
///
/// [coercions]: ../../book/deref-coercions.html
///
/// # Examples
///
/// A struct with a single field which is modifiable via dereferencing the
/// struct.
///
/// ```
/// use std::ops::{Deref, DerefMut};
///
/// struct DerefMutExample<T> {
///     value: T
/// }
///
/// impl<T> Deref for DerefMutExample<T> {
///     type Target = T;
///
///     fn deref<'a>(&'a self) -> &'a T {
///         &self.value
///     }
/// }
///
/// impl<T> DerefMut for DerefMutExample<T> {
///     fn deref_mut<'a>(&'a mut self) -> &'a mut T {
///         &mut self.value
///     }
/// }
///
/// fn main() {
///     let mut x = DerefMutExample { value: 'a' };
///     *x = 'b';
///     assert_eq!('b', *x);
/// }
/// ```
#[lang = "deref_mut"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DerefMut: Deref {
    /// The method called to mutably dereference a value
    #[stable(feature = "rust1", since = "1.0.0")]
    fn deref_mut(&mut self) -> &mut Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> DerefMut for &'a mut T {
    fn deref_mut(&mut self) -> &mut T { *self }
}

/// A version of the call operator that takes an immutable receiver.
#[lang = "fn"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait Fn<Args> : FnMut<Args> {
    /// This is called when the call operator is used.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

/// A version of the call operator that takes a mutable receiver.
#[lang = "fn_mut"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait FnMut<Args> : FnOnce<Args> {
    /// This is called when the call operator is used.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

/// A version of the call operator that takes a by-value receiver.
#[lang = "fn_once"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait FnOnce<Args> {
    /// The returned type after the call operator is used.
    #[unstable(feature = "fn_traits", issue = "29625")]
    type Output;

    /// This is called when the call operator is used.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

mod impls {
    use marker::Sized;
    use super::{Fn, FnMut, FnOnce};

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> Fn<A> for &'a F
        where F : Fn<A>
    {
        extern "rust-call" fn call(&self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnMut<A> for &'a F
        where F : Fn<A>
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnOnce<A> for &'a F
        where F : Fn<A>
    {
        type Output = F::Output;

        extern "rust-call" fn call_once(self, args: A) -> F::Output {
            (*self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnMut<A> for &'a mut F
        where F : FnMut<A>
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnOnce<A> for &'a mut F
        where F : FnMut<A>
    {
        type Output = F::Output;
        extern "rust-call" fn call_once(mut self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }
}

/// Trait that indicates that this is a pointer or a wrapper for one,
/// where unsizing can be performed on the pointee.
#[unstable(feature = "coerce_unsized", issue = "27732")]
#[lang="coerce_unsized"]
pub trait CoerceUnsized<T> {
    // Empty.
}

// &mut T -> &mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
// &mut T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}
// &mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}
// &mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}

// &T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
// &T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a T {}

// *mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
// *mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}

// *const T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}

/// Both `in (PLACE) EXPR` and `box EXPR` desugar into expressions
/// that allocate an intermediate "place" that holds uninitialized
/// state.  The desugaring evaluates EXPR, and writes the result at
/// the address returned by the `pointer` method of this trait.
///
/// A `Place` can be thought of as a special representation for a
/// hypothetical `&uninit` reference (which Rust cannot currently
/// express directly). That is, it represents a pointer to
/// uninitialized storage.
///
/// The client is responsible for two steps: First, initializing the
/// payload (it can access its address via `pointer`). Second,
/// converting the agent to an instance of the owning pointer, via the
/// appropriate `finalize` method (see the `InPlace`.
///
/// If evaluating EXPR fails, then the destructor for the
/// implementation of Place to clean up any intermediate state
/// (e.g. deallocate box storage, pop a stack, etc).
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Place<Data: ?Sized> {
    /// Returns the address where the input value will be written.
    /// Note that the data at this address is generally uninitialized,
    /// and thus one should use `ptr::write` for initializing it.
    fn pointer(&mut self) -> *mut Data;
}

/// Interface to implementations of  `in (PLACE) EXPR`.
///
/// `in (PLACE) EXPR` effectively desugars into:
///
/// ```rust,ignore
/// let p = PLACE;
/// let mut place = Placer::make_place(p);
/// let raw_place = Place::pointer(&mut place);
/// let value = EXPR;
/// unsafe {
///     std::ptr::write(raw_place, value);
///     InPlace::finalize(place)
/// }
/// ```
///
/// The type of `in (PLACE) EXPR` is derived from the type of `PLACE`;
/// if the type of `PLACE` is `P`, then the final type of the whole
/// expression is `P::Place::Owner` (see the `InPlace` and `Boxed`
/// traits).
///
/// Values for types implementing this trait usually are transient
/// intermediate values (e.g. the return value of `Vec::emplace_back`)
/// or `Copy`, since the `make_place` method takes `self` by value.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Placer<Data: ?Sized> {
    /// `Place` is the intermedate agent guarding the
    /// uninitialized state for `Data`.
    type Place: InPlace<Data>;

    /// Creates a fresh place from `self`.
    fn make_place(self) -> Self::Place;
}

/// Specialization of `Place` trait supporting `in (PLACE) EXPR`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait InPlace<Data: ?Sized>: Place<Data> {
    /// `Owner` is the type of the end value of `in (PLACE) EXPR`
    ///
    /// Note that when `in (PLACE) EXPR` is solely used for
    /// side-effecting an existing data-structure,
    /// e.g. `Vec::emplace_back`, then `Owner` need not carry any
    /// information at all (e.g. it can be the unit type `()` in that
    /// case).
    type Owner;

    /// Converts self into the final value, shifting
    /// deallocation/cleanup responsibilities (if any remain), over to
    /// the returned instance of `Owner` and forgetting self.
    unsafe fn finalize(self) -> Self::Owner;
}

/// Core trait for the `box EXPR` form.
///
/// `box EXPR` effectively desugars into:
///
/// ```rust,ignore
/// let mut place = BoxPlace::make_place();
/// let raw_place = Place::pointer(&mut place);
/// let value = EXPR;
/// unsafe {
///     ::std::ptr::write(raw_place, value);
///     Boxed::finalize(place)
/// }
/// ```
///
/// The type of `box EXPR` is supplied from its surrounding
/// context; in the above expansion, the result type `T` is used
/// to determine which implementation of `Boxed` to use, and that
/// `<T as Boxed>` in turn dictates determines which
/// implementation of `BoxPlace` to use, namely:
/// `<<T as Boxed>::Place as BoxPlace>`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Boxed {
    /// The kind of data that is stored in this kind of box.
    type Data;  /* (`Data` unused b/c cannot yet express below bound.) */
    /// The place that will negotiate the storage of the data.
    type Place: BoxPlace<Self::Data>;

    /// Converts filled place into final owning value, shifting
    /// deallocation/cleanup responsibilities (if any remain), over to
    /// returned instance of `Self` and forgetting `filled`.
    unsafe fn finalize(filled: Self::Place) -> Self;
}

/// Specialization of `Place` trait supporting `box EXPR`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait BoxPlace<Data: ?Sized> : Place<Data> {
    /// Creates a globally fresh place.
    fn make_place() -> Self;
}
