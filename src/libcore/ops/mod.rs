// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Overloadable operators.
//!
//! Implementing these traits allows you to overload certain operators.
//!
//! Some of these traits are imported by the prelude, so they are available in
//! every Rust program. Only operators backed by traits can be overloaded. For
//! example, the addition operator (`+`) can be overloaded through the [`Add`]
//! trait, but since the assignment operator (`=`) has no backing trait, there
//! is no way of overloading its semantics. Additionally, this module does not
//! provide any mechanism to create new operators. If traitless overloading or
//! custom operators are required, you should look toward macros or compiler
//! plugins to extend Rust's syntax.
//!
//! Note that the `&&` and `||` operators short-circuit, i.e. they only
//! evaluate their second operand if it contributes to the result. Since this
//! behavior is not enforceable by traits, `&&` and `||` are not supported as
//! overloadable operators.
//!
//! Many of the operators take their operands by value. In non-generic
//! contexts involving built-in types, this is usually not a problem.
//! However, using these operators in generic code, requires some
//! attention if values have to be reused as opposed to letting the operators
//! consume them. One option is to occasionally use [`clone`].
//! Another option is to rely on the types involved providing additional
//! operator implementations for references. For example, for a user-defined
//! type `T` which is supposed to support addition, it is probably a good
//! idea to have both `T` and `&T` implement the traits [`Add<T>`][`Add`] and
//! [`Add<&T>`][`Add`] so that generic code can be written without unnecessary
//! cloning.
//!
//! # Examples
//!
//! This example creates a `Point` struct that implements [`Add`] and [`Sub`],
//! and then demonstrates adding and subtracting two `Point`s.
//!
//! ```rust
//! use std::ops::{Add, Sub};
//!
//! #[derive(Debug)]
//! struct Point {
//!     x: i32,
//!     y: i32,
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
//! See the documentation for each trait for an example implementation.
//!
//! The [`Fn`], [`FnMut`], and [`FnOnce`] traits are implemented by types that can be
//! invoked like functions. Note that [`Fn`] takes `&self`, [`FnMut`] takes `&mut
//! self` and [`FnOnce`] takes `self`. These correspond to the three kinds of
//! methods that can be invoked on an instance: call-by-reference,
//! call-by-mutable-reference, and call-by-value. The most common use of these
//! traits is to act as bounds to higher-level functions that take functions or
//! closures as arguments.
//!
//! Taking a [`Fn`] as a parameter:
//!
//! ```rust
//! fn call_with_one<F>(func: F) -> usize
//!     where F: Fn(usize) -> usize
//! {
//!     func(1)
//! }
//!
//! let double = |x| x * 2;
//! assert_eq!(call_with_one(double), 2);
//! ```
//!
//! Taking a [`FnMut`] as a parameter:
//!
//! ```rust
//! fn do_twice<F>(mut func: F)
//!     where F: FnMut()
//! {
//!     func();
//!     func();
//! }
//!
//! let mut x: usize = 1;
//! {
//!     let add_two_to_x = || x += 2;
//!     do_twice(add_two_to_x);
//! }
//!
//! assert_eq!(x, 5);
//! ```
//!
//! Taking a [`FnOnce`] as a parameter:
//!
//! ```rust
//! fn consume_with_relish<F>(func: F)
//!     where F: FnOnce() -> String
//! {
//!     // `func` consumes its captured variables, so it cannot be run more
//!     // than once
//!     println!("Consumed: {}", func());
//!
//!     println!("Delicious!");
//!
//!     // Attempting to invoke `func()` again will throw a `use of moved
//!     // value` error for `func`
//! }
//!
//! let x = String::from("x");
//! let consume_and_return_x = move || x;
//! consume_with_relish(consume_and_return_x);
//!
//! // `consume_and_return_x` can no longer be invoked at this point
//! ```
//!
//! [`Fn`]: trait.Fn.html
//! [`FnMut`]: trait.FnMut.html
//! [`FnOnce`]: trait.FnOnce.html
//! [`Add`]: trait.Add.html
//! [`Sub`]: trait.Sub.html
//! [`clone`]: ../clone/trait.Clone.html#tymethod.clone

#![stable(feature = "rust1", since = "1.0.0")]

use fmt;
use marker::Unsize;

/// The `Drop` trait is used to run some code when a value goes out of scope.
/// This is sometimes called a 'destructor'.
///
/// When a value goes out of scope, if it implements this trait, it will have
/// its `drop` method called. Then any fields the value contains will also
/// be dropped recursively.
///
/// Because of the recursive dropping, you do not need to implement this trait
/// unless your type needs its own destructor logic.
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
///
/// Showing the recursive nature of `Drop`. When `outer` goes out of scope, the
/// `drop` method will be called first for `Outer`, then for `Inner`. Therefore
/// `main` prints `Dropping Outer!` and then `Dropping Inner!`.
///
/// ```
/// struct Inner;
/// struct Outer(Inner);
///
/// impl Drop for Inner {
///     fn drop(&mut self) {
///         println!("Dropping Inner!");
///     }
/// }
///
/// impl Drop for Outer {
///     fn drop(&mut self) {
///         println!("Dropping Outer!");
///     }
/// }
///
/// fn main() {
///     let _x = Outer(Inner);
/// }
/// ```
///
/// Because variables are dropped in the reverse order they are declared,
/// `main` will print `Declared second!` and then `Declared first!`.
///
/// ```
/// struct PrintOnDrop(&'static str);
///
/// fn main() {
///     let _first = PrintOnDrop("Declared first!");
///     let _second = PrintOnDrop("Declared second!");
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
    /// This function cannot be called explicitly. This is compiler error
    /// [E0040]. However, the [`std::mem::drop`] function in the prelude can be
    /// used to call the argument's `Drop` implementation.
    ///
    /// [E0040]: ../../error-index.html#E0040
    /// [`std::mem::drop`]: ../../std/mem/fn.drop.html
    ///
    /// # Panics
    ///
    /// Given that a `panic!` will call `drop()` as it unwinds, any `panic!` in
    /// a `drop()` implementation will likely abort.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}

/// The addition operator `+`.
///
/// # Examples
///
/// This example creates a `Point` struct that implements the `Add` trait, and
/// then demonstrates adding two `Point`s.
///
/// ```
/// use std::ops::Add;
///
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Add for Point {
///     type Output = Point;
///
///     fn add(self, other: Point) -> Point {
///         Point {
///             x: self.x + other.x,
///             y: self.y + other.y,
///         }
///     }
/// }
///
/// impl PartialEq for Point {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// fn main() {
///     assert_eq!(Point { x: 1, y: 0 } + Point { x: 2, y: 3 },
///                Point { x: 3, y: 3 });
/// }
/// ```
///
/// Here is an example of the same `Point` struct implementing the `Add` trait
/// using generics.
///
/// ```
/// use std::ops::Add;
///
/// #[derive(Debug)]
/// struct Point<T> {
///     x: T,
///     y: T,
/// }
///
/// // Notice that the implementation uses the `Output` associated type
/// impl<T: Add<Output=T>> Add for Point<T> {
///     type Output = Point<T>;
///
///     fn add(self, other: Point<T>) -> Point<T> {
///         Point {
///             x: self.x + other.x,
///             y: self.y + other.y,
///         }
///     }
/// }
///
/// impl<T: PartialEq> PartialEq for Point<T> {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// fn main() {
///     assert_eq!(Point { x: 1, y: 0 } + Point { x: 2, y: 3 },
///                Point { x: 3, y: 3 });
/// }
/// ```
///
/// Note that `RHS = Self` by default, but this is not mandatory. For example,
/// [std::time::SystemTime] implements `Add<Duration>`, which permits
/// operations of the form `SystemTime = SystemTime + Duration`.
///
/// [std::time::SystemTime]: ../../std/time/struct.SystemTime.html
#[lang = "add"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} + {RHS}`"]
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
            #[rustc_inherit_overflow_checks]
            fn add(self, other: $t) -> $t { self + other }
        }

        forward_ref_binop! { impl Add, add for $t, $t }
    )*)
}

add_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The subtraction operator `-`.
///
/// # Examples
///
/// This example creates a `Point` struct that implements the `Sub` trait, and
/// then demonstrates subtracting two `Point`s.
///
/// ```
/// use std::ops::Sub;
///
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Sub for Point {
///     type Output = Point;
///
///     fn sub(self, other: Point) -> Point {
///         Point {
///             x: self.x - other.x,
///             y: self.y - other.y,
///         }
///     }
/// }
///
/// impl PartialEq for Point {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// fn main() {
///     assert_eq!(Point { x: 3, y: 3 } - Point { x: 2, y: 3 },
///                Point { x: 1, y: 0 });
/// }
/// ```
///
/// Note that `RHS = Self` by default, but this is not mandatory. For example,
/// [std::time::SystemTime] implements `Sub<Duration>`, which permits
/// operations of the form `SystemTime = SystemTime - Duration`.
///
/// [std::time::SystemTime]: ../../std/time/struct.SystemTime.html
#[lang = "sub"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} - {RHS}`"]
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
            #[rustc_inherit_overflow_checks]
            fn sub(self, other: $t) -> $t { self - other }
        }

        forward_ref_binop! { impl Sub, sub for $t, $t }
    )*)
}

sub_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The multiplication operator `*`.
///
/// # Examples
///
/// Implementing a `Mul`tipliable rational number struct:
///
/// ```
/// use std::ops::Mul;
///
/// // The uniqueness of rational numbers in lowest terms is a consequence of
/// // the fundamental theorem of arithmetic.
/// #[derive(Eq)]
/// #[derive(PartialEq, Debug)]
/// struct Rational {
///     nominator: usize,
///     denominator: usize,
/// }
///
/// impl Rational {
///     fn new(nominator: usize, denominator: usize) -> Self {
///         if denominator == 0 {
///             panic!("Zero is an invalid denominator!");
///         }
///
///         // Reduce to lowest terms by dividing by the greatest common
///         // divisor.
///         let gcd = gcd(nominator, denominator);
///         Rational {
///             nominator: nominator / gcd,
///             denominator: denominator / gcd,
///         }
///     }
/// }
///
/// impl Mul for Rational {
///     // The multiplication of rational numbers is a closed operation.
///     type Output = Self;
///
///     fn mul(self, rhs: Self) -> Self {
///         let nominator = self.nominator * rhs.nominator;
///         let denominator = self.denominator * rhs.denominator;
///         Rational::new(nominator, denominator)
///     }
/// }
///
/// // Euclid's two-thousand-year-old algorithm for finding the greatest common
/// // divisor.
/// fn gcd(x: usize, y: usize) -> usize {
///     let mut x = x;
///     let mut y = y;
///     while y != 0 {
///         let t = y;
///         y = x % y;
///         x = t;
///     }
///     x
/// }
///
/// assert_eq!(Rational::new(1, 2), Rational::new(2, 4));
/// assert_eq!(Rational::new(2, 3) * Rational::new(3, 4),
///            Rational::new(1, 2));
/// ```
///
/// Note that `RHS = Self` by default, but this is not mandatory. Here is an
/// implementation which enables multiplication of vectors by scalars, as is
/// done in linear algebra.
///
/// ```
/// use std::ops::Mul;
///
/// struct Scalar {value: usize};
///
/// #[derive(Debug)]
/// struct Vector {value: Vec<usize>};
///
/// impl Mul<Vector> for Scalar {
///     type Output = Vector;
///
///     fn mul(self, rhs: Vector) -> Vector {
///         Vector {value: rhs.value.iter().map(|v| self.value * v).collect()}
///     }
/// }
///
/// impl PartialEq<Vector> for Vector {
///     fn eq(&self, other: &Self) -> bool {
///         self.value == other.value
///     }
/// }
///
/// let scalar = Scalar{value: 3};
/// let vector = Vector{value: vec![2, 4, 6]};
/// assert_eq!(scalar * vector, Vector{value: vec![6, 12, 18]});
/// ```
#[lang = "mul"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} * {RHS}`"]
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
            #[rustc_inherit_overflow_checks]
            fn mul(self, other: $t) -> $t { self * other }
        }

        forward_ref_binop! { impl Mul, mul for $t, $t }
    )*)
}

mul_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The division operator `/`.
///
/// # Examples
///
/// Implementing a `Div`idable rational number struct:
///
/// ```
/// use std::ops::Div;
///
/// // The uniqueness of rational numbers in lowest terms is a consequence of
/// // the fundamental theorem of arithmetic.
/// #[derive(Eq)]
/// #[derive(PartialEq, Debug)]
/// struct Rational {
///     nominator: usize,
///     denominator: usize,
/// }
///
/// impl Rational {
///     fn new(nominator: usize, denominator: usize) -> Self {
///         if denominator == 0 {
///             panic!("Zero is an invalid denominator!");
///         }
///
///         // Reduce to lowest terms by dividing by the greatest common
///         // divisor.
///         let gcd = gcd(nominator, denominator);
///         Rational {
///             nominator: nominator / gcd,
///             denominator: denominator / gcd,
///         }
///     }
/// }
///
/// impl Div for Rational {
///     // The division of rational numbers is a closed operation.
///     type Output = Self;
///
///     fn div(self, rhs: Self) -> Self {
///         if rhs.nominator == 0 {
///             panic!("Cannot divide by zero-valued `Rational`!");
///         }
///
///         let nominator = self.nominator * rhs.denominator;
///         let denominator = self.denominator * rhs.nominator;
///         Rational::new(nominator, denominator)
///     }
/// }
///
/// // Euclid's two-thousand-year-old algorithm for finding the greatest common
/// // divisor.
/// fn gcd(x: usize, y: usize) -> usize {
///     let mut x = x;
///     let mut y = y;
///     while y != 0 {
///         let t = y;
///         y = x % y;
///         x = t;
///     }
///     x
/// }
///
/// fn main() {
///     assert_eq!(Rational::new(1, 2), Rational::new(2, 4));
///     assert_eq!(Rational::new(1, 2) / Rational::new(3, 4),
///                Rational::new(2, 3));
/// }
/// ```
///
/// Note that `RHS = Self` by default, but this is not mandatory. Here is an
/// implementation which enables division of vectors by scalars, as is done in
/// linear algebra.
///
/// ```
/// use std::ops::Div;
///
/// struct Scalar {value: f32};
///
/// #[derive(Debug)]
/// struct Vector {value: Vec<f32>};
///
/// impl Div<Scalar> for Vector {
///     type Output = Vector;
///
///     fn div(self, rhs: Scalar) -> Vector {
///         Vector {value: self.value.iter().map(|v| v / rhs.value).collect()}
///     }
/// }
///
/// impl PartialEq<Vector> for Vector {
///     fn eq(&self, other: &Self) -> bool {
///         self.value == other.value
///     }
/// }
///
/// let scalar = Scalar{value: 2f32};
/// let vector = Vector{value: vec![2f32, 4f32, 6f32]};
/// assert_eq!(vector / scalar, Vector{value: vec![1f32, 2f32, 3f32]});
/// ```
#[lang = "div"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} / {RHS}`"]
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

div_impl_integer! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

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

/// The remainder operator `%`.
///
/// # Examples
///
/// This example implements `Rem` on a `SplitSlice` object. After `Rem` is
/// implemented, one can use the `%` operator to find out what the remaining
/// elements of the slice would be after splitting it into equal slices of a
/// given length.
///
/// ```
/// use std::ops::Rem;
///
/// #[derive(PartialEq, Debug)]
/// struct SplitSlice<'a, T: 'a> {
///     slice: &'a [T],
/// }
///
/// impl<'a, T> Rem<usize> for SplitSlice<'a, T> {
///     type Output = SplitSlice<'a, T>;
///
///     fn rem(self, modulus: usize) -> Self {
///         let len = self.slice.len();
///         let rem = len % modulus;
///         let start = len - rem;
///         SplitSlice {slice: &self.slice[start..]}
///     }
/// }
///
/// // If we were to divide &[0, 1, 2, 3, 4, 5, 6, 7] into slices of size 3,
/// // the remainder would be &[6, 7]
/// assert_eq!(SplitSlice { slice: &[0, 1, 2, 3, 4, 5, 6, 7] } % 3,
///            SplitSlice { slice: &[6, 7] });
/// ```
#[lang = "rem"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} % {RHS}`"]
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

rem_impl_integer! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }


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

/// The unary negation operator `-`.
///
/// # Examples
///
/// An implementation of `Neg` for `Sign`, which allows the use of `-` to
/// negate its value.
///
/// ```
/// use std::ops::Neg;
///
/// #[derive(Debug, PartialEq)]
/// enum Sign {
///     Negative,
///     Zero,
///     Positive,
/// }
///
/// impl Neg for Sign {
///     type Output = Sign;
///
///     fn neg(self) -> Sign {
///         match self {
///             Sign::Negative => Sign::Positive,
///             Sign::Zero => Sign::Zero,
///             Sign::Positive => Sign::Negative,
///         }
///     }
/// }
///
/// // a negative positive is a negative
/// assert_eq!(-Sign::Positive, Sign::Negative);
/// // a double negative is a positive
/// assert_eq!(-Sign::Negative, Sign::Positive);
/// // zero is its own negation
/// assert_eq!(-Sign::Zero, Sign::Zero);
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
            #[rustc_inherit_overflow_checks]
            fn neg(self) -> $t { let $id = self; $body }
        }

        forward_ref_unop! { impl Neg, neg for $t }
    )*)
}

macro_rules! neg_impl_numeric {
    ($($t:ty)*) => { neg_impl_core!{ x => -x, $($t)*} }
}

#[allow(unused_macros)]
macro_rules! neg_impl_unsigned {
    ($($t:ty)*) => {
        neg_impl_core!{ x => {
            !x.wrapping_add(1)
        }, $($t)*} }
}

// neg_impl_unsigned! { usize u8 u16 u32 u64 }
neg_impl_numeric! { isize i8 i16 i32 i64 i128 f32 f64 }

/// The unary logical negation operator `!`.
///
/// # Examples
///
/// An implementation of `Not` for `Answer`, which enables the use of `!` to
/// invert its value.
///
/// ```
/// use std::ops::Not;
///
/// #[derive(Debug, PartialEq)]
/// enum Answer {
///     Yes,
///     No,
/// }
///
/// impl Not for Answer {
///     type Output = Answer;
///
///     fn not(self) -> Answer {
///         match self {
///             Answer::Yes => Answer::No,
///             Answer::No => Answer::Yes
///         }
///     }
/// }
///
/// assert_eq!(!Answer::Yes, Answer::No);
/// assert_eq!(!Answer::No, Answer::Yes);
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

not_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise AND operator `&`.
///
/// # Examples
///
/// In this example, the `&` operator is lifted to a trivial `Scalar` type.
///
/// ```
/// use std::ops::BitAnd;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(bool);
///
/// impl BitAnd for Scalar {
///     type Output = Self;
///
///     // rhs is the "right-hand side" of the expression `a & b`
///     fn bitand(self, rhs: Self) -> Self {
///         Scalar(self.0 & rhs.0)
///     }
/// }
///
/// fn main() {
///     assert_eq!(Scalar(true) & Scalar(true), Scalar(true));
///     assert_eq!(Scalar(true) & Scalar(false), Scalar(false));
///     assert_eq!(Scalar(false) & Scalar(true), Scalar(false));
///     assert_eq!(Scalar(false) & Scalar(false), Scalar(false));
/// }
/// ```
///
/// In this example, the `BitAnd` trait is implemented for a `BooleanVector`
/// struct.
///
/// ```
/// use std::ops::BitAnd;
///
/// #[derive(Debug, PartialEq)]
/// struct BooleanVector(Vec<bool>);
///
/// impl BitAnd for BooleanVector {
///     type Output = Self;
///
///     fn bitand(self, BooleanVector(rhs): Self) -> Self {
///         let BooleanVector(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         BooleanVector(lhs.iter().zip(rhs.iter()).map(|(x, y)| *x && *y).collect())
///     }
/// }
///
/// fn main() {
///     let bv1 = BooleanVector(vec![true, true, false, false]);
///     let bv2 = BooleanVector(vec![true, false, true, false]);
///     let expected = BooleanVector(vec![true, false, false, false]);
///     assert_eq!(bv1 & bv2, expected);
/// }
/// ```
#[lang = "bitand"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} & {RHS}`"]
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

bitand_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise OR operator `|`.
///
/// # Examples
///
/// In this example, the `|` operator is lifted to a trivial `Scalar` type.
///
/// ```
/// use std::ops::BitOr;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(bool);
///
/// impl BitOr for Scalar {
///     type Output = Self;
///
///     // rhs is the "right-hand side" of the expression `a | b`
///     fn bitor(self, rhs: Self) -> Self {
///         Scalar(self.0 | rhs.0)
///     }
/// }
///
/// fn main() {
///     assert_eq!(Scalar(true) | Scalar(true), Scalar(true));
///     assert_eq!(Scalar(true) | Scalar(false), Scalar(true));
///     assert_eq!(Scalar(false) | Scalar(true), Scalar(true));
///     assert_eq!(Scalar(false) | Scalar(false), Scalar(false));
/// }
/// ```
///
/// In this example, the `BitOr` trait is implemented for a `BooleanVector`
/// struct.
///
/// ```
/// use std::ops::BitOr;
///
/// #[derive(Debug, PartialEq)]
/// struct BooleanVector(Vec<bool>);
///
/// impl BitOr for BooleanVector {
///     type Output = Self;
///
///     fn bitor(self, BooleanVector(rhs): Self) -> Self {
///         let BooleanVector(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         BooleanVector(lhs.iter().zip(rhs.iter()).map(|(x, y)| *x || *y).collect())
///     }
/// }
///
/// fn main() {
///     let bv1 = BooleanVector(vec![true, true, false, false]);
///     let bv2 = BooleanVector(vec![true, false, true, false]);
///     let expected = BooleanVector(vec![true, true, true, false]);
///     assert_eq!(bv1 | bv2, expected);
/// }
/// ```
#[lang = "bitor"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} | {RHS}`"]
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

bitor_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise XOR operator `^`.
///
/// # Examples
///
/// In this example, the `^` operator is lifted to a trivial `Scalar` type.
///
/// ```
/// use std::ops::BitXor;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(bool);
///
/// impl BitXor for Scalar {
///     type Output = Self;
///
///     // rhs is the "right-hand side" of the expression `a ^ b`
///     fn bitxor(self, rhs: Self) -> Self {
///         Scalar(self.0 ^ rhs.0)
///     }
/// }
///
/// fn main() {
///     assert_eq!(Scalar(true) ^ Scalar(true), Scalar(false));
///     assert_eq!(Scalar(true) ^ Scalar(false), Scalar(true));
///     assert_eq!(Scalar(false) ^ Scalar(true), Scalar(true));
///     assert_eq!(Scalar(false) ^ Scalar(false), Scalar(false));
/// }
/// ```
///
/// In this example, the `BitXor` trait is implemented for a `BooleanVector`
/// struct.
///
/// ```
/// use std::ops::BitXor;
///
/// #[derive(Debug, PartialEq)]
/// struct BooleanVector(Vec<bool>);
///
/// impl BitXor for BooleanVector {
///     type Output = Self;
///
///     fn bitxor(self, BooleanVector(rhs): Self) -> Self {
///         let BooleanVector(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         BooleanVector(lhs.iter()
///                          .zip(rhs.iter())
///                          .map(|(x, y)| (*x || *y) && !(*x && *y))
///                          .collect())
///     }
/// }
///
/// fn main() {
///     let bv1 = BooleanVector(vec![true, true, false, false]);
///     let bv2 = BooleanVector(vec![true, false, true, false]);
///     let expected = BooleanVector(vec![false, true, true, false]);
///     assert_eq!(bv1 ^ bv2, expected);
/// }
/// ```
#[lang = "bitxor"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} ^ {RHS}`"]
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

bitxor_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The left shift operator `<<`.
///
/// # Examples
///
/// An implementation of `Shl` that lifts the `<<` operation on integers to a
/// `Scalar` struct.
///
/// ```
/// use std::ops::Shl;
///
/// #[derive(PartialEq, Debug)]
/// struct Scalar(usize);
///
/// impl Shl<Scalar> for Scalar {
///     type Output = Self;
///
///     fn shl(self, Scalar(rhs): Self) -> Scalar {
///         let Scalar(lhs) = self;
///         Scalar(lhs << rhs)
///     }
/// }
/// fn main() {
///     assert_eq!(Scalar(4) << Scalar(2), Scalar(16));
/// }
/// ```
///
/// An implementation of `Shl` that spins a vector leftward by a given amount.
///
/// ```
/// use std::ops::Shl;
///
/// #[derive(PartialEq, Debug)]
/// struct SpinVector<T: Clone> {
///     vec: Vec<T>,
/// }
///
/// impl<T: Clone> Shl<usize> for SpinVector<T> {
///     type Output = Self;
///
///     fn shl(self, rhs: usize) -> SpinVector<T> {
///         // rotate the vector by `rhs` places
///         let (a, b) = self.vec.split_at(rhs);
///         let mut spun_vector: Vec<T> = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         SpinVector { vec: spun_vector }
///     }
/// }
///
/// fn main() {
///     assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } << 2,
///                SpinVector { vec: vec![2, 3, 4, 0, 1] });
/// }
/// ```
#[lang = "shl"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} << {RHS}`"]
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
            #[rustc_inherit_overflow_checks]
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
        shl_impl! { $t, u128 }
        shl_impl! { $t, usize }

        shl_impl! { $t, i8 }
        shl_impl! { $t, i16 }
        shl_impl! { $t, i32 }
        shl_impl! { $t, i64 }
        shl_impl! { $t, i128 }
        shl_impl! { $t, isize }
    )*)
}

shl_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 isize i128 }

/// The right shift operator `>>`.
///
/// # Examples
///
/// An implementation of `Shr` that lifts the `>>` operation on integers to a
/// `Scalar` struct.
///
/// ```
/// use std::ops::Shr;
///
/// #[derive(PartialEq, Debug)]
/// struct Scalar(usize);
///
/// impl Shr<Scalar> for Scalar {
///     type Output = Self;
///
///     fn shr(self, Scalar(rhs): Self) -> Scalar {
///         let Scalar(lhs) = self;
///         Scalar(lhs >> rhs)
///     }
/// }
/// fn main() {
///     assert_eq!(Scalar(16) >> Scalar(2), Scalar(4));
/// }
/// ```
///
/// An implementation of `Shr` that spins a vector rightward by a given amount.
///
/// ```
/// use std::ops::Shr;
///
/// #[derive(PartialEq, Debug)]
/// struct SpinVector<T: Clone> {
///     vec: Vec<T>,
/// }
///
/// impl<T: Clone> Shr<usize> for SpinVector<T> {
///     type Output = Self;
///
///     fn shr(self, rhs: usize) -> SpinVector<T> {
///         // rotate the vector by `rhs` places
///         let (a, b) = self.vec.split_at(self.vec.len() - rhs);
///         let mut spun_vector: Vec<T> = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         SpinVector { vec: spun_vector }
///     }
/// }
///
/// fn main() {
///     assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } >> 2,
///                SpinVector { vec: vec![3, 4, 0, 1, 2] });
/// }
/// ```
#[lang = "shr"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} >> {RHS}`"]
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
            #[rustc_inherit_overflow_checks]
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
        shr_impl! { $t, u128 }
        shr_impl! { $t, usize }

        shr_impl! { $t, i8 }
        shr_impl! { $t, i16 }
        shr_impl! { $t, i32 }
        shr_impl! { $t, i64 }
        shr_impl! { $t, i128 }
        shr_impl! { $t, isize }
    )*)
}

shr_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

/// The addition assignment operator `+=`.
///
/// # Examples
///
/// This example creates a `Point` struct that implements the `AddAssign`
/// trait, and then demonstrates add-assigning to a mutable `Point`.
///
/// ```
/// use std::ops::AddAssign;
///
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl AddAssign for Point {
///     fn add_assign(&mut self, other: Point) {
///         *self = Point {
///             x: self.x + other.x,
///             y: self.y + other.y,
///         };
///     }
/// }
///
/// impl PartialEq for Point {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// let mut point = Point { x: 1, y: 0 };
/// point += Point { x: 2, y: 3 };
/// assert_eq!(point, Point { x: 3, y: 3 });
/// ```
#[lang = "add_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} += {Rhs}`"]
pub trait AddAssign<Rhs=Self> {
    /// The method for the `+=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn add_assign(&mut self, rhs: Rhs);
}

macro_rules! add_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl AddAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn add_assign(&mut self, other: $t) { *self += other }
        }
    )+)
}

add_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The subtraction assignment operator `-=`.
///
/// # Examples
///
/// This example creates a `Point` struct that implements the `SubAssign`
/// trait, and then demonstrates sub-assigning to a mutable `Point`.
///
/// ```
/// use std::ops::SubAssign;
///
/// #[derive(Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl SubAssign for Point {
///     fn sub_assign(&mut self, other: Point) {
///         *self = Point {
///             x: self.x - other.x,
///             y: self.y - other.y,
///         };
///     }
/// }
///
/// impl PartialEq for Point {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// let mut point = Point { x: 3, y: 3 };
/// point -= Point { x: 2, y: 3 };
/// assert_eq!(point, Point {x: 1, y: 0});
/// ```
#[lang = "sub_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} -= {Rhs}`"]
pub trait SubAssign<Rhs=Self> {
    /// The method for the `-=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn sub_assign(&mut self, rhs: Rhs);
}

macro_rules! sub_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl SubAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn sub_assign(&mut self, other: $t) { *self -= other }
        }
    )+)
}

sub_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The multiplication assignment operator `*=`.
///
/// # Examples
///
/// A trivial implementation of `MulAssign`. When `Foo *= Foo` happens, it ends up
/// calling `mul_assign`, and therefore, `main` prints `Multiplying!`.
///
/// ```
/// use std::ops::MulAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} *= {Rhs}`"]
pub trait MulAssign<Rhs=Self> {
    /// The method for the `*=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn mul_assign(&mut self, rhs: Rhs);
}

macro_rules! mul_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl MulAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn mul_assign(&mut self, other: $t) { *self *= other }
        }
    )+)
}

mul_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The division assignment operator `/=`.
///
/// # Examples
///
/// A trivial implementation of `DivAssign`. When `Foo /= Foo` happens, it ends up
/// calling `div_assign`, and therefore, `main` prints `Dividing!`.
///
/// ```
/// use std::ops::DivAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} /= {Rhs}`"]
pub trait DivAssign<Rhs=Self> {
    /// The method for the `/=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn div_assign(&mut self, rhs: Rhs);
}

macro_rules! div_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl DivAssign for $t {
            #[inline]
            fn div_assign(&mut self, other: $t) { *self /= other }
        }
    )+)
}

div_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The remainder assignment operator `%=`.
///
/// # Examples
///
/// A trivial implementation of `RemAssign`. When `Foo %= Foo` happens, it ends up
/// calling `rem_assign`, and therefore, `main` prints `Remainder-ing!`.
///
/// ```
/// use std::ops::RemAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} %= {Rhs}`"]
pub trait RemAssign<Rhs=Self> {
    /// The method for the `%=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn rem_assign(&mut self, rhs: Rhs);
}

macro_rules! rem_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl RemAssign for $t {
            #[inline]
            fn rem_assign(&mut self, other: $t) { *self %= other }
        }
    )+)
}

rem_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

/// The bitwise AND assignment operator `&=`.
///
/// # Examples
///
/// In this example, the `&=` operator is lifted to a trivial `Scalar` type.
///
/// ```
/// use std::ops::BitAndAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(bool);
///
/// impl BitAndAssign for Scalar {
///     // rhs is the "right-hand side" of the expression `a &= b`
///     fn bitand_assign(&mut self, rhs: Self) {
///         *self = Scalar(self.0 & rhs.0)
///     }
/// }
///
/// fn main() {
///     let mut scalar = Scalar(true);
///     scalar &= Scalar(true);
///     assert_eq!(scalar, Scalar(true));
///
///     let mut scalar = Scalar(true);
///     scalar &= Scalar(false);
///     assert_eq!(scalar, Scalar(false));
///
///     let mut scalar = Scalar(false);
///     scalar &= Scalar(true);
///     assert_eq!(scalar, Scalar(false));
///
///     let mut scalar = Scalar(false);
///     scalar &= Scalar(false);
///     assert_eq!(scalar, Scalar(false));
/// }
/// ```
///
/// In this example, the `BitAndAssign` trait is implemented for a
/// `BooleanVector` struct.
///
/// ```
/// use std::ops::BitAndAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct BooleanVector(Vec<bool>);
///
/// impl BitAndAssign for BooleanVector {
///     // rhs is the "right-hand side" of the expression `a &= b`
///     fn bitand_assign(&mut self, rhs: Self) {
///         assert_eq!(self.0.len(), rhs.0.len());
///         *self = BooleanVector(self.0
///                                   .iter()
///                                   .zip(rhs.0.iter())
///                                   .map(|(x, y)| *x && *y)
///                                   .collect());
///     }
/// }
///
/// fn main() {
///     let mut bv = BooleanVector(vec![true, true, false, false]);
///     bv &= BooleanVector(vec![true, false, true, false]);
///     let expected = BooleanVector(vec![true, false, false, false]);
///     assert_eq!(bv, expected);
/// }
/// ```
#[lang = "bitand_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} &= {Rhs}`"]
pub trait BitAndAssign<Rhs=Self> {
    /// The method for the `&=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitand_assign(&mut self, rhs: Rhs);
}

macro_rules! bitand_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitAndAssign for $t {
            #[inline]
            fn bitand_assign(&mut self, other: $t) { *self &= other }
        }
    )+)
}

bitand_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise OR assignment operator `|=`.
///
/// # Examples
///
/// A trivial implementation of `BitOrAssign`. When `Foo |= Foo` happens, it ends up
/// calling `bitor_assign`, and therefore, `main` prints `Bitwise Or-ing!`.
///
/// ```
/// use std::ops::BitOrAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} |= {Rhs}`"]
pub trait BitOrAssign<Rhs=Self> {
    /// The method for the `|=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitor_assign(&mut self, rhs: Rhs);
}

macro_rules! bitor_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitOrAssign for $t {
            #[inline]
            fn bitor_assign(&mut self, other: $t) { *self |= other }
        }
    )+)
}

bitor_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise XOR assignment operator `^=`.
///
/// # Examples
///
/// A trivial implementation of `BitXorAssign`. When `Foo ^= Foo` happens, it ends up
/// calling `bitxor_assign`, and therefore, `main` prints `Bitwise Xor-ing!`.
///
/// ```
/// use std::ops::BitXorAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} ^= {Rhs}`"]
pub trait BitXorAssign<Rhs=Self> {
    /// The method for the `^=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitxor_assign(&mut self, rhs: Rhs);
}

macro_rules! bitxor_assign_impl {
    ($($t:ty)+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitXorAssign for $t {
            #[inline]
            fn bitxor_assign(&mut self, other: $t) { *self ^= other }
        }
    )+)
}

bitxor_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The left shift assignment operator `<<=`.
///
/// # Examples
///
/// A trivial implementation of `ShlAssign`. When `Foo <<= Foo` happens, it ends up
/// calling `shl_assign`, and therefore, `main` prints `Shifting left!`.
///
/// ```
/// use std::ops::ShlAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} <<= {Rhs}`"]
pub trait ShlAssign<Rhs> {
    /// The method for the `<<=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn shl_assign(&mut self, rhs: Rhs);
}

macro_rules! shl_assign_impl {
    ($t:ty, $f:ty) => (
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShlAssign<$f> for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
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
        shl_assign_impl! { $t, u128 }
        shl_assign_impl! { $t, usize }

        shl_assign_impl! { $t, i8 }
        shl_assign_impl! { $t, i16 }
        shl_assign_impl! { $t, i32 }
        shl_assign_impl! { $t, i64 }
        shl_assign_impl! { $t, i128 }
        shl_assign_impl! { $t, isize }
    )*)
}

shl_assign_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

/// The right shift assignment operator `>>=`.
///
/// # Examples
///
/// A trivial implementation of `ShrAssign`. When `Foo >>= Foo` happens, it ends up
/// calling `shr_assign`, and therefore, `main` prints `Shifting right!`.
///
/// ```
/// use std::ops::ShrAssign;
///
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
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented = "no implementation for `{Self} >>= {Rhs}`"]
pub trait ShrAssign<Rhs=Self> {
    /// The method for the `>>=` operator
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn shr_assign(&mut self, rhs: Rhs);
}

macro_rules! shr_assign_impl {
    ($t:ty, $f:ty) => (
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
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
        shr_assign_impl! { $t, u128 }
        shr_assign_impl! { $t, usize }

        shr_assign_impl! { $t, i8 }
        shr_assign_impl! { $t, i16 }
        shr_assign_impl! { $t, i32 }
        shr_assign_impl! { $t, i64 }
        shr_assign_impl! { $t, i128 }
        shr_assign_impl! { $t, isize }
    )*)
}

shr_assign_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

/// The `Index` trait is used to specify the functionality of indexing operations
/// like `container[index]` when used in an immutable context.
///
/// `container[index]` is actually syntactic sugar for `*container.index(index)`,
/// but only when used as an immutable value. If a mutable value is requested,
/// [`IndexMut`] is used instead. This allows nice things such as
/// `let value = v[index]` if `value` implements [`Copy`].
///
/// [`IndexMut`]: ../../std/ops/trait.IndexMut.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
///
/// # Examples
///
/// The following example implements `Index` on a read-only `NucleotideCount`
/// container, enabling individual counts to be retrieved with index syntax.
///
/// ```
/// use std::ops::Index;
///
/// enum Nucleotide {
///     A,
///     C,
///     G,
///     T,
/// }
///
/// struct NucleotideCount {
///     a: usize,
///     c: usize,
///     g: usize,
///     t: usize,
/// }
///
/// impl Index<Nucleotide> for NucleotideCount {
///     type Output = usize;
///
///     fn index(&self, nucleotide: Nucleotide) -> &usize {
///         match nucleotide {
///             Nucleotide::A => &self.a,
///             Nucleotide::C => &self.c,
///             Nucleotide::G => &self.g,
///             Nucleotide::T => &self.t,
///         }
///     }
/// }
///
/// let nucleotide_count = NucleotideCount {a: 14, c: 9, g: 10, t: 12};
/// assert_eq!(nucleotide_count[Nucleotide::A], 14);
/// assert_eq!(nucleotide_count[Nucleotide::C], 9);
/// assert_eq!(nucleotide_count[Nucleotide::G], 10);
/// assert_eq!(nucleotide_count[Nucleotide::T], 12);
/// ```
#[lang = "index"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Index<Idx: ?Sized> {
    /// The returned type after indexing
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output: ?Sized;

    /// The method for the indexing (`container[index]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index(&self, index: Idx) -> &Self::Output;
}

/// The `IndexMut` trait is used to specify the functionality of indexing
/// operations like `container[index]` when used in a mutable context.
///
/// `container[index]` is actually syntactic sugar for
/// `*container.index_mut(index)`, but only when used as a mutable value. If
/// an immutable value is requested, the [`Index`] trait is used instead. This
/// allows nice things such as `v[index] = value` if `value` implements [`Copy`].
///
/// [`Index`]: ../../std/ops/trait.Index.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
///
/// # Examples
///
/// A very simple implementation of a `Balance` struct that has two sides, where
/// each can be indexed mutably and immutably.
///
/// ```
/// use std::ops::{Index,IndexMut};
///
/// #[derive(Debug)]
/// enum Side {
///     Left,
///     Right,
/// }
///
/// #[derive(Debug, PartialEq)]
/// enum Weight {
///     Kilogram(f32),
///     Pound(f32),
/// }
///
/// struct Balance {
///     pub left: Weight,
///     pub right:Weight,
/// }
///
/// impl Index<Side> for Balance {
///     type Output = Weight;
///
///     fn index<'a>(&'a self, index: Side) -> &'a Weight {
///         println!("Accessing {:?}-side of balance immutably", index);
///         match index {
///             Side::Left => &self.left,
///             Side::Right => &self.right,
///         }
///     }
/// }
///
/// impl IndexMut<Side> for Balance {
///     fn index_mut<'a>(&'a mut self, index: Side) -> &'a mut Weight {
///         println!("Accessing {:?}-side of balance mutably", index);
///         match index {
///             Side::Left => &mut self.left,
///             Side::Right => &mut self.right,
///         }
///     }
/// }
///
/// fn main() {
///     let mut balance = Balance {
///         right: Weight::Kilogram(2.5),
///         left: Weight::Pound(1.5),
///     };
///
///     // In this case balance[Side::Right] is sugar for
///     // *balance.index(Side::Right), since we are only reading
///     // balance[Side::Right], not writing it.
///     assert_eq!(balance[Side::Right],Weight::Kilogram(2.5));
///
///     // However in this case balance[Side::Left] is sugar for
///     // *balance.index_mut(Side::Left), since we are writing
///     // balance[Side::Left].
///     balance[Side::Left] = Weight::Kilogram(3.0);
/// }
/// ```
#[lang = "index_mut"]
#[rustc_on_unimplemented = "the type `{Self}` cannot be mutably indexed by `{Idx}`"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    /// The method for the mutable indexing (`container[index]`) operation
    #[stable(feature = "rust1", since = "1.0.0")]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}

/// An unbounded range. Use `..` (two dots) for its shorthand.
///
/// Its primary use case is slicing index. It cannot serve as an iterator
/// because it doesn't have a starting point.
///
/// # Examples
///
/// The `..` syntax is a `RangeFull`:
///
/// ```
/// assert_eq!((..), std::ops::RangeFull);
/// ```
///
/// It does not have an `IntoIterator` implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```ignore
/// for i in .. {
///    // ...
/// }
/// ```
///
/// Used as a slicing index, `RangeFull` produces the full array as a slice.
///
/// ```
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);  // RangeFull
/// assert_eq!(arr[ ..3], [0,1,2  ]);
/// assert_eq!(arr[1.. ], [  1,2,3]);
/// assert_eq!(arr[1..3], [  1,2  ]);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RangeFull;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for RangeFull {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "..")
    }
}

/// A (half-open) range which is bounded at both ends: { x | start <= x < end }.
/// Use `start..end` (two dots) for its shorthand.
///
/// See the [`contains`](#method.contains) method for its characterization.
///
/// # Examples
///
/// ```
/// fn main() {
///     assert_eq!((3..5), std::ops::Range{ start: 3, end: 5 });
///     assert_eq!(3+4+5, (3..6).sum());
///
///     let arr = [0, 1, 2, 3];
///     assert_eq!(arr[ .. ], [0,1,2,3]);
///     assert_eq!(arr[ ..3], [0,1,2  ]);
///     assert_eq!(arr[1.. ], [  1,2,3]);
///     assert_eq!(arr[1..3], [  1,2  ]);  // Range
/// }
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
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

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> Range<Idx> {
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    /// fn main() {
    ///     assert!( ! (3..5).contains(2));
    ///     assert!(   (3..5).contains(3));
    ///     assert!(   (3..5).contains(4));
    ///     assert!( ! (3..5).contains(5));
    ///
    ///     assert!( ! (3..3).contains(3));
    ///     assert!( ! (3..2).contains(3));
    /// }
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (self.start <= item) && (item < self.end)
    }
}

/// A range which is only bounded below: { x | start <= x }.
/// Use `start..` for its shorthand.
///
/// See the [`contains`](#method.contains) method for its characterization.
///
/// Note: Currently, no overflow checking is done for the iterator
/// implementation; if you use an integer range and the integer overflows, it
/// might panic in debug mode or create an endless loop in release mode. This
/// overflow behavior might change in the future.
///
/// # Examples
///
/// ```
/// fn main() {
///     assert_eq!((2..), std::ops::RangeFrom{ start: 2 });
///     assert_eq!(2+3+4, (2..).take(3).sum());
///
///     let arr = [0, 1, 2, 3];
///     assert_eq!(arr[ .. ], [0,1,2,3]);
///     assert_eq!(arr[ ..3], [0,1,2  ]);
///     assert_eq!(arr[1.. ], [  1,2,3]);  // RangeFrom
///     assert_eq!(arr[1..3], [  1,2  ]);
/// }
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
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

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeFrom<Idx> {
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    /// fn main() {
    ///     assert!( ! (3..).contains(2));
    ///     assert!(   (3..).contains(3));
    ///     assert!(   (3..).contains(1_000_000_000));
    /// }
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (self.start <= item)
    }
}

/// A range which is only bounded above: { x | x < end }.
/// Use `..end` (two dots) for its shorthand.
///
/// See the [`contains`](#method.contains) method for its characterization.
///
/// It cannot serve as an iterator because it doesn't have a starting point.
///
/// # Examples
///
/// The `..{integer}` syntax is a `RangeTo`:
///
/// ```
/// assert_eq!((..5), std::ops::RangeTo{ end: 5 });
/// ```
///
/// It does not have an `IntoIterator` implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```ignore
/// for i in ..5 {
///     // ...
/// }
/// ```
///
/// When used as a slicing index, `RangeTo` produces a slice of all array
/// elements before the index indicated by `end`.
///
/// ```
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ .. ], [0,1,2,3]);
/// assert_eq!(arr[ ..3], [0,1,2  ]);  // RangeTo
/// assert_eq!(arr[1.. ], [  1,2,3]);
/// assert_eq!(arr[1..3], [  1,2  ]);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
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

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeTo<Idx> {
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains)]
    /// fn main() {
    ///     assert!(   (..5).contains(-1_000_000_000));
    ///     assert!(   (..5).contains(4));
    ///     assert!( ! (..5).contains(5));
    /// }
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (item < self.end)
    }
}

/// An inclusive range which is bounded at both ends: { x | start <= x <= end }.
/// Use `start...end` (three dots) for its shorthand.
///
/// See the [`contains`](#method.contains) method for its characterization.
///
/// # Examples
///
/// ```
/// #![feature(inclusive_range,inclusive_range_syntax)]
/// fn main() {
///     assert_eq!((3...5), std::ops::RangeInclusive{ start: 3, end: 5 });
///     assert_eq!(3+4+5, (3...5).sum());
///
///     let arr = [0, 1, 2, 3];
///     assert_eq!(arr[ ...2], [0,1,2  ]);
///     assert_eq!(arr[1...2], [  1,2  ]);  // RangeInclusive
/// }
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]  // not Copy -- see #27186
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
pub struct RangeInclusive<Idx> {
    /// The lower bound of the range (inclusive).
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub start: Idx,
    /// The upper bound of the range (inclusive).
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub end: Idx,
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<Idx: fmt::Debug> fmt::Debug for RangeInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}...{:?}", self.start, self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeInclusive<Idx> {
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains,inclusive_range_syntax)]
    /// fn main() {
    ///     assert!( ! (3...5).contains(2));
    ///     assert!(   (3...5).contains(3));
    ///     assert!(   (3...5).contains(4));
    ///     assert!(   (3...5).contains(5));
    ///     assert!( ! (3...5).contains(6));
    ///
    ///     assert!(   (3...3).contains(3));
    ///     assert!( ! (3...2).contains(3));
    /// }
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        self.start <= item && item <= self.end
    }
}

/// An inclusive range which is only bounded above: { x | x <= end }.
/// Use `...end` (three dots) for its shorthand.
///
/// See the [`contains`](#method.contains) method for its characterization.
///
/// It cannot serve as an iterator because it doesn't have a starting point.
///
/// # Examples
///
/// The `...{integer}` syntax is a `RangeToInclusive`:
///
/// ```
/// #![feature(inclusive_range,inclusive_range_syntax)]
/// assert_eq!((...5), std::ops::RangeToInclusive{ end: 5 });
/// ```
///
/// It does not have an `IntoIterator` implementation, so you can't use it in a
/// `for` loop directly. This won't compile:
///
/// ```ignore
/// for i in ...5 {
///     // ...
/// }
/// ```
///
/// When used as a slicing index, `RangeToInclusive` produces a slice of all
/// array elements up to and including the index indicated by `end`.
///
/// ```
/// #![feature(inclusive_range_syntax)]
/// let arr = [0, 1, 2, 3];
/// assert_eq!(arr[ ...2], [0,1,2  ]);  // RangeToInclusive
/// assert_eq!(arr[1...2], [  1,2  ]);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
pub struct RangeToInclusive<Idx> {
    /// The upper bound of the range (inclusive)
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    pub end: Idx,
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<Idx: fmt::Debug> fmt::Debug for RangeToInclusive<Idx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "...{:?}", self.end)
    }
}

#[unstable(feature = "range_contains", reason = "recently added as per RFC", issue = "32311")]
impl<Idx: PartialOrd<Idx>> RangeToInclusive<Idx> {
    /// # Examples
    ///
    /// ```
    /// #![feature(range_contains,inclusive_range_syntax)]
    /// fn main() {
    ///     assert!(   (...5).contains(-1_000_000_000));
    ///     assert!(   (...5).contains(5));
    ///     assert!( ! (...5).contains(6));
    /// }
    /// ```
    pub fn contains(&self, item: Idx) -> bool {
        (item <= self.end)
    }
}

// RangeToInclusive<Idx> cannot impl From<RangeTo<Idx>>
// because underflow would be possible with (..0).into()

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
///     fn deref(&self) -> &T {
///         &self.value
///     }
/// }
///
/// impl<T> DerefMut for DerefMutExample<T> {
///     fn deref_mut(&mut self) -> &mut T {
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
///
/// # Examples
///
/// Closures automatically implement this trait, which allows them to be
/// invoked. Note, however, that `Fn` takes an immutable reference to any
/// captured variables. To take a mutable capture, implement [`FnMut`], and to
/// consume the capture, implement [`FnOnce`].
///
/// [`FnMut`]: trait.FnMut.html
/// [`FnOnce`]: trait.FnOnce.html
///
/// ```
/// let square = |x| x * x;
/// assert_eq!(square(5), 25);
/// ```
///
/// Closures can also be passed to higher-level functions through a `Fn`
/// parameter (or a `FnMut` or `FnOnce` parameter, which are supertraits of
/// `Fn`).
///
/// ```
/// fn call_with_one<F>(func: F) -> usize
///     where F: Fn(usize) -> usize {
///     func(1)
/// }
///
/// let double = |x| x * 2;
/// assert_eq!(call_with_one(double), 2);
/// ```
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
///
/// # Examples
///
/// Closures that mutably capture variables automatically implement this trait,
/// which allows them to be invoked.
///
/// ```
/// let mut x = 5;
/// {
///     let mut square_x = || x *= x;
///     square_x();
/// }
/// assert_eq!(x, 25);
/// ```
///
/// Closures can also be passed to higher-level functions through a `FnMut`
/// parameter (or a `FnOnce` parameter, which is a supertrait of `FnMut`).
///
/// ```
/// fn do_twice<F>(mut func: F)
///     where F: FnMut()
/// {
///     func();
///     func();
/// }
///
/// let mut x: usize = 1;
/// {
///     let add_two_to_x = || x += 2;
///     do_twice(add_two_to_x);
/// }
///
/// assert_eq!(x, 5);
/// ```
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
///
/// # Examples
///
/// By-value closures automatically implement this trait, which allows them to
/// be invoked.
///
/// ```
/// let x = 5;
/// let square_x = move || x * x;
/// assert_eq!(square_x(), 25);
/// ```
///
/// By-value Closures can also be passed to higher-level functions through a
/// `FnOnce` parameter.
///
/// ```
/// fn consume_with_relish<F>(func: F)
///     where F: FnOnce() -> String
/// {
///     // `func` consumes its captured variables, so it cannot be run more
///     // than once
///     println!("Consumed: {}", func());
///
///     println!("Delicious!");
///
///     // Attempting to invoke `func()` again will throw a `use of moved
///     // value` error for `func`
/// }
///
/// let x = String::from("x");
/// let consume_and_return_x = move || x;
/// consume_with_relish(consume_and_return_x);
///
/// // `consume_and_return_x` can no longer be invoked at this point
/// ```
#[lang = "fn_once"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait FnOnce<Args> {
    /// The returned type after the call operator is used.
    #[stable(feature = "fn_once_output", since = "1.12.0")]
    type Output;

    /// This is called when the call operator is used.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

mod impls {
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
///
/// See the [DST coercion RfC][dst-coerce] and [the nomicon entry on coercion][nomicon-coerce]
/// for more details.
///
/// For builtin pointer types, pointers to `T` will coerce to pointers to `U` if `T: Unsize<U>`
/// by converting from a thin pointer to a fat pointer.
///
/// For custom types, the coercion here works by coercing `Foo<T>` to `Foo<U>`
/// provided an impl of `CoerceUnsized<Foo<U>> for Foo<T>` exists.
/// Such an impl can only be written if `Foo<T>` has only a single non-phantomdata
/// field involving `T`. If the type of that field is `Bar<T>`, an implementation
/// of `CoerceUnsized<Bar<U>> for Bar<T>` must exist. The coercion will work by
/// by coercing the `Bar<T>` field into `Bar<U>` and filling in the rest of the fields
/// from `Foo<T>` to create a `Foo<U>`. This will effectively drill down to a pointer
/// field and coerce that.
///
/// Generally, for smart pointers you will implement
/// `CoerceUnsized<Ptr<U>> for Ptr<T> where T: Unsize<U>, U: ?Sized`, with an
/// optional `?Sized` bound on `T` itself. For wrapper types that directly embed `T`
/// like `Cell<T>` and `RefCell<T>`, you
/// can directly implement `CoerceUnsized<Wrap<U>> for Wrap<T> where T: CoerceUnsized<U>`.
/// This will let coercions of types like `Cell<Box<T>>` work.
///
/// [`Unsize`][unsize] is used to mark types which can be coerced to DSTs if behind
/// pointers. It is implemented automatically by the compiler.
///
/// [dst-coerce]: https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md
/// [unsize]: ../marker/trait.Unsize.html
/// [nomicon-coerce]: ../../nomicon/coercions.html
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

/// Both `PLACE <- EXPR` and `box EXPR` desugar into expressions
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
/// If evaluating EXPR fails, then it is up to the destructor for the
/// implementation of Place to clean up any intermediate state
/// (e.g. deallocate box storage, pop a stack, etc).
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Place<Data: ?Sized> {
    /// Returns the address where the input value will be written.
    /// Note that the data at this address is generally uninitialized,
    /// and thus one should use `ptr::write` for initializing it.
    fn pointer(&mut self) -> *mut Data;
}

/// Interface to implementations of  `PLACE <- EXPR`.
///
/// `PLACE <- EXPR` effectively desugars into:
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
/// The type of `PLACE <- EXPR` is derived from the type of `PLACE`;
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

/// Specialization of `Place` trait supporting `PLACE <- EXPR`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait InPlace<Data: ?Sized>: Place<Data> {
    /// `Owner` is the type of the end value of `PLACE <- EXPR`
    ///
    /// Note that when `PLACE <- EXPR` is solely used for
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

/// This trait has been superseded by the `Try` trait, but must remain
/// here as `?` is still lowered to it in stage0 .
#[cfg(stage0)]
#[unstable(feature = "question_mark_carrier", issue = "31436")]
pub trait Carrier {
    /// The type of the value when computation succeeds.
    type Success;
    /// The type of the value when computation errors out.
    type Error;

    /// Create a `Carrier` from a success value.
    fn from_success(_: Self::Success) -> Self;

    /// Create a `Carrier` from an error value.
    fn from_error(_: Self::Error) -> Self;

    /// Translate this `Carrier` to another implementation of `Carrier` with the
    /// same associated types.
    fn translate<T>(self) -> T where T: Carrier<Success=Self::Success, Error=Self::Error>;
}

#[cfg(stage0)]
#[unstable(feature = "question_mark_carrier", issue = "31436")]
impl<U, V> Carrier for Result<U, V> {
    type Success = U;
    type Error = V;

    fn from_success(u: U) -> Result<U, V> {
        Ok(u)
    }

    fn from_error(e: V) -> Result<U, V> {
        Err(e)
    }

    fn translate<T>(self) -> T
        where T: Carrier<Success=U, Error=V>
    {
        match self {
            Ok(u) => T::from_success(u),
            Err(e) => T::from_error(e),
        }
    }
}

struct _DummyErrorType;

impl Try for _DummyErrorType {
    type Ok = ();
    type Error = ();

    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn from_ok(_: ()) -> _DummyErrorType {
        _DummyErrorType
    }

    fn from_error(_: ()) -> _DummyErrorType {
        _DummyErrorType
    }
}

/// A trait for customizing the behaviour of the `?` operator.
///
/// A type implementing `Try` is one that has a canonical way to view it
/// in terms of a success/failure dichotomy.  This trait allows both
/// extracting those success or failure values from an existing instance and
/// creating a new instance from a success or failure value.
#[unstable(feature = "try_trait", issue = "42327")]
pub trait Try {
    /// The type of this value when viewed as successful.
    #[unstable(feature = "try_trait", issue = "42327")]
    type Ok;
    /// The type of this value when viewed as failed.
    #[unstable(feature = "try_trait", issue = "42327")]
    type Error;

    /// Applies the "?" operator. A return of `Ok(t)` means that the
    /// execution should continue normally, and the result of `?` is the
    /// value `t`. A return of `Err(e)` means that execution should branch
    /// to the innermost enclosing `catch`, or return from the function.
    ///
    /// If an `Err(e)` result is returned, the value `e` will be "wrapped"
    /// in the return type of the enclosing scope (which must itself implement
    /// `Try`). Specifically, the value `X::from_error(From::from(e))`
    /// is returned, where `X` is the return type of the enclosing function.
    #[unstable(feature = "try_trait", issue = "42327")]
    fn into_result(self) -> Result<Self::Ok, Self::Error>;

    /// Wrap an error value to construct the composite result. For example,
    /// `Result::Err(x)` and `Result::from_error(x)` are equivalent.
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_error(v: Self::Error) -> Self;

    /// Wrap an OK value to construct the composite result. For example,
    /// `Result::Ok(x)` and `Result::from_ok(x)` are equivalent.
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_ok(v: Self::Ok) -> Self;
}
