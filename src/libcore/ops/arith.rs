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
/// Here is an example of the same `Point` struct implementing the `Sub` trait
/// using generics.
///
/// ```
/// use std::ops::Sub;
///
/// #[derive(Debug)]
/// struct Point<T> {
///     x: T,
///     y: T,
/// }
///
/// // Notice that the implementation uses the `Output` associated type
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
/// impl<T: PartialEq> PartialEq for Point<T> {
///     fn eq(&self, other: &Self) -> bool {
///         self.x == other.x && self.y == other.y
///     }
/// }
///
/// fn main() {
///     assert_eq!(Point { x: 2, y: 3 } - Point { x: 1, y: 0 },
///                Point { x: 1, y: 3 });
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
