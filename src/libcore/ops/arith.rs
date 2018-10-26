// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The addition operator `+`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory. For
/// example, [`std::time::SystemTime`] implements `Add<Duration>`, which permits
/// operations of the form `SystemTime = SystemTime + Duration`.
///
/// [`std::time::SystemTime`]: ../../std/time/struct.SystemTime.html
///
/// # Examples
///
/// ## `Add`able points
///
/// ```
/// use std::ops::Add;
///
/// #[derive(Debug, PartialEq)]
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
/// assert_eq!(Point { x: 1, y: 0 } + Point { x: 2, y: 3 },
///            Point { x: 3, y: 3 });
/// ```
///
/// ## Implementing `Add` with generics
///
/// Here is an example of the same `Point` struct implementing the `Add` trait
/// using generics.
///
/// ```
/// use std::ops::Add;
///
/// #[derive(Debug, PartialEq)]
/// struct Point<T> {
///     x: T,
///     y: T,
/// }
///
/// // Notice that the implementation uses the associated type `Output`.
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
/// assert_eq!(Point { x: 1, y: 0 } + Point { x: 2, y: 3 },
///            Point { x: 3, y: 3 });
/// ```
#[lang = "add"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    on(
        all(_Self="{integer}", RHS="{float}"),
        message="cannot add a float to an integer",
    ),
    on(
        all(_Self="{float}", RHS="{integer}"),
        message="cannot add an integer to a float",
    ),
    message="cannot add `{RHS}` to `{Self}`",
    label="no implementation for `{Self} + {RHS}`",
)]
#[doc(alias = "+")]
pub trait Add<RHS=Self> {
    /// The resulting type after applying the `+` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `+` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn add(self, rhs: RHS) -> Self::Output;
}

macro_rules! add_impl {
    ($($t:ty), *) => ($(
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

add_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The subtraction operator `-`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory. For
/// example, [`std::time::SystemTime`] implements `Sub<Duration>`, which permits
/// operations of the form `SystemTime = SystemTime - Duration`.
///
/// [`std::time::SystemTime`]: ../../std/time/struct.SystemTime.html
///
/// # Examples
///
/// ## `Sub`tractable points
///
/// ```
/// use std::ops::Sub;
///
/// #[derive(Debug, PartialEq)]
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
/// assert_eq!(Point { x: 3, y: 3 } - Point { x: 2, y: 3 },
///            Point { x: 1, y: 0 });
/// ```
///
/// ## Implementing `Sub` with generics
///
/// Here is an example of the same `Point` struct implementing the `Sub` trait
/// using generics.
///
/// ```
/// use std::ops::Sub;
///
/// #[derive(Debug, PartialEq)]
/// struct Point<T> {
///     x: T,
///     y: T,
/// }
///
/// // Notice that the implementation uses the associated type `Output`.
/// impl<T: Sub<Output=T>> Sub for Point<T> {
///     type Output = Point<T>;
///
///     fn sub(self, other: Point<T>) -> Point<T> {
///         Point {
///             x: self.x - other.x,
///             y: self.y - other.y,
///         }
///     }
/// }
///
/// assert_eq!(Point { x: 2, y: 3 } - Point { x: 1, y: 0 },
///            Point { x: 1, y: 3 });
/// ```
#[lang = "sub"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="cannot subtract `{RHS}` from `{Self}`",
                         label="no implementation for `{Self} - {RHS}`")]
#[doc(alias = "-")]
pub trait Sub<RHS=Self> {
    /// The resulting type after applying the `-` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `-` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sub(self, rhs: RHS) -> Self::Output;
}

macro_rules! sub_impl {
    ($($t:ty),*) => ($(
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

sub_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The multiplication operator `*`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
///
/// # Examples
///
/// ## `Mul`tipliable rational numbers
///
/// ```
/// use std::ops::Mul;
///
/// // By the fundamental theorem of arithmetic, rational numbers in lowest
/// // terms are unique. So, by keeping `Rational`s in reduced form, we can
/// // derive `Eq` and `PartialEq`.
/// #[derive(Debug, Eq, PartialEq)]
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
/// ## Multiplying vectors by scalars as in linear algebra
///
/// ```
/// use std::ops::Mul;
///
/// struct Scalar { value: usize }
///
/// #[derive(Debug, PartialEq)]
/// struct Vector { value: Vec<usize> }
///
/// impl Mul<Scalar> for Vector {
///     type Output = Vector;
///
///     fn mul(self, rhs: Scalar) -> Vector {
///         Vector { value: self.value.iter().map(|v| v * rhs.value).collect() }
///     }
/// }
///
/// let vector = Vector { value: vec![2, 4, 6] };
/// let scalar = Scalar { value: 3 };
/// assert_eq!(vector * scalar, Vector { value: vec![6, 12, 18] });
/// ```
#[lang = "mul"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="cannot multiply `{RHS}` to `{Self}`",
                         label="no implementation for `{Self} * {RHS}`")]
#[doc(alias = "*")]
pub trait Mul<RHS=Self> {
    /// The resulting type after applying the `*` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `*` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn mul(self, rhs: RHS) -> Self::Output;
}

macro_rules! mul_impl {
    ($($t:ty),*) => ($(
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

mul_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The division operator `/`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
///
/// # Examples
///
/// ## `Div`idable rational numbers
///
/// ```
/// use std::ops::Div;
///
/// // By the fundamental theorem of arithmetic, rational numbers in lowest
/// // terms are unique. So, by keeping `Rational`s in reduced form, we can
/// // derive `Eq` and `PartialEq`.
/// #[derive(Debug, Eq, PartialEq)]
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
/// assert_eq!(Rational::new(1, 2), Rational::new(2, 4));
/// assert_eq!(Rational::new(1, 2) / Rational::new(3, 4),
///            Rational::new(2, 3));
/// ```
///
/// ## Dividing vectors by scalars as in linear algebra
///
/// ```
/// use std::ops::Div;
///
/// struct Scalar { value: f32 }
///
/// #[derive(Debug, PartialEq)]
/// struct Vector { value: Vec<f32> }
///
/// impl Div<Scalar> for Vector {
///     type Output = Vector;
///
///     fn div(self, rhs: Scalar) -> Vector {
///         Vector { value: self.value.iter().map(|v| v / rhs.value).collect() }
///     }
/// }
///
/// let scalar = Scalar { value: 2f32 };
/// let vector = Vector { value: vec![2f32, 4f32, 6f32] };
/// assert_eq!(vector / scalar, Vector { value: vec![1f32, 2f32, 3f32] });
/// ```
#[lang = "div"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="cannot divide `{Self}` by `{RHS}`",
                         label="no implementation for `{Self} / {RHS}`")]
#[doc(alias = "/")]
pub trait Div<RHS=Self> {
    /// The resulting type after applying the `/` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `/` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn div(self, rhs: RHS) -> Self::Output;
}

macro_rules! div_impl_integer {
    ($($t:ty),*) => ($(
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

div_impl_integer! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

macro_rules! div_impl_float {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Div for $t {
            type Output = $t;

            #[inline]
            fn div(self, other: $t) -> $t { self / other }
        }

        forward_ref_binop! { impl Div, div for $t, $t }
    )*)
}

div_impl_float! { f32, f64 }

/// The remainder operator `%`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
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
/// // the remainder would be &[6, 7].
/// assert_eq!(SplitSlice { slice: &[0, 1, 2, 3, 4, 5, 6, 7] } % 3,
///            SplitSlice { slice: &[6, 7] });
/// ```
#[lang = "rem"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="cannot mod `{Self}` by `{RHS}`",
                         label="no implementation for `{Self} % {RHS}`")]
#[doc(alias = "%")]
pub trait Rem<RHS=Self> {
    /// The resulting type after applying the `%` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output = Self;

    /// Performs the `%` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rem(self, rhs: RHS) -> Self::Output;
}

macro_rules! rem_impl_integer {
    ($($t:ty),*) => ($(
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

rem_impl_integer! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

macro_rules! rem_impl_float {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Rem for $t {
            type Output = $t;

            #[inline]
            fn rem(self, other: $t) -> $t { self % other }
        }

        forward_ref_binop! { impl Rem, rem for $t, $t }
    )*)
}

rem_impl_float! { f32, f64 }

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
/// // A negative positive is a negative.
/// assert_eq!(-Sign::Positive, Sign::Negative);
/// // A double negative is a positive.
/// assert_eq!(-Sign::Negative, Sign::Positive);
/// // Zero is its own negation.
/// assert_eq!(-Sign::Zero, Sign::Zero);
/// ```
#[lang = "neg"]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = "-")]
pub trait Neg {
    /// The resulting type after applying the `-` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the unary `-` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn neg(self) -> Self::Output;
}



macro_rules! neg_impl_core {
    ($id:ident => $body:expr, $($t:ty),*) => ($(
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
    ($($t:ty),*) => { neg_impl_core!{ x => -x, $($t),*} }
}

#[allow(unused_macros)]
macro_rules! neg_impl_unsigned {
    ($($t:ty),*) => {
        neg_impl_core!{ x => {
            !x.wrapping_add(1)
        }, $($t),*} }
}

// neg_impl_unsigned! { usize u8 u16 u32 u64 }
neg_impl_numeric! { isize, i8, i16, i32, i64, i128, f32, f64 }

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
/// #[derive(Debug, PartialEq)]
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
/// let mut point = Point { x: 1, y: 0 };
/// point += Point { x: 2, y: 3 };
/// assert_eq!(point, Point { x: 3, y: 3 });
/// ```
#[lang = "add_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented(message="cannot add-assign `{Rhs}` to `{Self}`",
                         label="no implementation for `{Self} += {Rhs}`")]
#[doc(alias = "+")]
#[doc(alias = "+=")]
pub trait AddAssign<Rhs=Self> {
    /// Performs the `+=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn add_assign(&mut self, rhs: Rhs);
}

macro_rules! add_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl AddAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn add_assign(&mut self, other: $t) { *self += other }
        }

        forward_ref_op_assign! { impl AddAssign, add_assign for $t, $t }
    )+)
}

add_assign_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

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
/// #[derive(Debug, PartialEq)]
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
/// let mut point = Point { x: 3, y: 3 };
/// point -= Point { x: 2, y: 3 };
/// assert_eq!(point, Point {x: 1, y: 0});
/// ```
#[lang = "sub_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented(message="cannot subtract-assign `{Rhs}` from `{Self}`",
                         label="no implementation for `{Self} -= {Rhs}`")]
#[doc(alias = "-")]
#[doc(alias = "-=")]
pub trait SubAssign<Rhs=Self> {
    /// Performs the `-=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn sub_assign(&mut self, rhs: Rhs);
}

macro_rules! sub_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl SubAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn sub_assign(&mut self, other: $t) { *self -= other }
        }

        forward_ref_op_assign! { impl SubAssign, sub_assign for $t, $t }
    )+)
}

sub_assign_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The multiplication assignment operator `*=`.
///
/// # Examples
///
/// ```
/// use std::ops::MulAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Frequency { hertz: f64 }
///
/// impl MulAssign<f64> for Frequency {
///     fn mul_assign(&mut self, rhs: f64) {
///         self.hertz *= rhs;
///     }
/// }
///
/// let mut frequency = Frequency { hertz: 50.0 };
/// frequency *= 4.0;
/// assert_eq!(Frequency { hertz: 200.0 }, frequency);
/// ```
#[lang = "mul_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented(message="cannot multiply-assign `{Rhs}` to `{Self}`",
                         label="no implementation for `{Self} *= {Rhs}`")]
#[doc(alias = "*")]
#[doc(alias = "*=")]
pub trait MulAssign<Rhs=Self> {
    /// Performs the `*=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn mul_assign(&mut self, rhs: Rhs);
}

macro_rules! mul_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl MulAssign for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn mul_assign(&mut self, other: $t) { *self *= other }
        }

        forward_ref_op_assign! { impl MulAssign, mul_assign for $t, $t }
    )+)
}

mul_assign_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The division assignment operator `/=`.
///
/// # Examples
///
/// ```
/// use std::ops::DivAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Frequency { hertz: f64 }
///
/// impl DivAssign<f64> for Frequency {
///     fn div_assign(&mut self, rhs: f64) {
///         self.hertz /= rhs;
///     }
/// }
///
/// let mut frequency = Frequency { hertz: 200.0 };
/// frequency /= 4.0;
/// assert_eq!(Frequency { hertz: 50.0 }, frequency);
/// ```
#[lang = "div_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented(message="cannot divide-assign `{Self}` by `{Rhs}`",
                         label="no implementation for `{Self} /= {Rhs}`")]
#[doc(alias = "/")]
#[doc(alias = "/=")]
pub trait DivAssign<Rhs=Self> {
    /// Performs the `/=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn div_assign(&mut self, rhs: Rhs);
}

macro_rules! div_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl DivAssign for $t {
            #[inline]
            fn div_assign(&mut self, other: $t) { *self /= other }
        }

        forward_ref_op_assign! { impl DivAssign, div_assign for $t, $t }
    )+)
}

div_assign_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }

/// The remainder assignment operator `%=`.
///
/// # Examples
///
/// ```
/// use std::ops::RemAssign;
///
/// struct CookieJar { cookies: u32 }
///
/// impl RemAssign<u32> for CookieJar {
///     fn rem_assign(&mut self, piles: u32) {
///         self.cookies %= piles;
///     }
/// }
///
/// let mut jar = CookieJar { cookies: 31 };
/// let piles = 4;
///
/// println!("Splitting up {} cookies into {} even piles!", jar.cookies, piles);
///
/// jar %= piles;
///
/// println!("{} cookies remain in the cookie jar!", jar.cookies);
/// ```
#[lang = "rem_assign"]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[rustc_on_unimplemented(message="cannot mod-assign `{Self}` by `{Rhs}``",
                         label="no implementation for `{Self} %= {Rhs}`")]
#[doc(alias = "%")]
#[doc(alias = "%=")]
pub trait RemAssign<Rhs=Self> {
    /// Performs the `%=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn rem_assign(&mut self, rhs: Rhs);
}

macro_rules! rem_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl RemAssign for $t {
            #[inline]
            fn rem_assign(&mut self, other: $t) { *self %= other }
        }

        forward_ref_op_assign! { impl RemAssign, rem_assign for $t, $t }
    )+)
}

rem_assign_impl! { usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64 }
