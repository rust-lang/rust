/// The addition operator `+`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory. For
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
/// #[derive(Debug, Copy, Clone, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Add for Point {
///     type Output = Self;
///
///     fn add(self, other: Self) -> Self {
///         Self {
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
/// #[derive(Debug, Copy, Clone, PartialEq)]
/// struct Point<T> {
///     x: T,
///     y: T,
/// }
///
/// // Notice that the implementation uses the associated type `Output`.
/// impl<T: Add<Output = T>> Add for Point<T> {
///     type Output = Self;
///
///     fn add(self, other: Self) -> Self::Output {
///         Self {
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
    on(all(_Self = "{integer}", Rhs = "{float}"), message = "cannot add a float to an integer",),
    on(all(_Self = "{float}", Rhs = "{integer}"), message = "cannot add an integer to a float",),
    message = "cannot add `{Rhs}` to `{Self}`",
    label = "no implementation for `{Self} + {Rhs}`",
    append_const_msg
)]
#[doc(alias = "+")]
pub trait Add<Rhs = Self> {
    /// The resulting type after applying the `+` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `+` operation.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(12 + 1, 13);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "add"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn add(self, rhs: Rhs) -> Self::Output;
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
/// Note that `Rhs` is `Self` by default, but this is not mandatory. For
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
/// #[derive(Debug, Copy, Clone, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl Sub for Point {
///     type Output = Self;
///
///     fn sub(self, other: Self) -> Self::Output {
///         Self {
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
/// impl<T: Sub<Output = T>> Sub for Point<T> {
///     type Output = Self;
///
///     fn sub(self, other: Self) -> Self::Output {
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
#[rustc_on_unimplemented(
    message = "cannot subtract `{Rhs}` from `{Self}`",
    label = "no implementation for `{Self} - {Rhs}`",
    append_const_msg
)]
#[doc(alias = "-")]
pub trait Sub<Rhs = Self> {
    /// The resulting type after applying the `-` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `-` operation.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(12 - 1, 11);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "sub"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn sub(self, rhs: Rhs) -> Self::Output;
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
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
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
///     numerator: usize,
///     denominator: usize,
/// }
///
/// impl Rational {
///     fn new(numerator: usize, denominator: usize) -> Self {
///         if denominator == 0 {
///             panic!("Zero is an invalid denominator!");
///         }
///
///         // Reduce to lowest terms by dividing by the greatest common
///         // divisor.
///         let gcd = gcd(numerator, denominator);
///         Self {
///             numerator: numerator / gcd,
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
///         let numerator = self.numerator * rhs.numerator;
///         let denominator = self.denominator * rhs.denominator;
///         Self::new(numerator, denominator)
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
///     type Output = Self;
///
///     fn mul(self, rhs: Scalar) -> Self::Output {
///         Self { value: self.value.iter().map(|v| v * rhs.value).collect() }
///     }
/// }
///
/// let vector = Vector { value: vec![2, 4, 6] };
/// let scalar = Scalar { value: 3 };
/// assert_eq!(vector * scalar, Vector { value: vec![6, 12, 18] });
/// ```
#[lang = "mul"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    message = "cannot multiply `{Self}` by `{Rhs}`",
    label = "no implementation for `{Self} * {Rhs}`"
)]
#[doc(alias = "*")]
pub trait Mul<Rhs = Self> {
    /// The resulting type after applying the `*` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `*` operation.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(12 * 2, 24);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "mul"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn mul(self, rhs: Rhs) -> Self::Output;
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
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
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
///     numerator: usize,
///     denominator: usize,
/// }
///
/// impl Rational {
///     fn new(numerator: usize, denominator: usize) -> Self {
///         if denominator == 0 {
///             panic!("Zero is an invalid denominator!");
///         }
///
///         // Reduce to lowest terms by dividing by the greatest common
///         // divisor.
///         let gcd = gcd(numerator, denominator);
///         Self {
///             numerator: numerator / gcd,
///             denominator: denominator / gcd,
///         }
///     }
/// }
///
/// impl Div for Rational {
///     // The division of rational numbers is a closed operation.
///     type Output = Self;
///
///     fn div(self, rhs: Self) -> Self::Output {
///         if rhs.numerator == 0 {
///             panic!("Cannot divide by zero-valued `Rational`!");
///         }
///
///         let numerator = self.numerator * rhs.denominator;
///         let denominator = self.denominator * rhs.numerator;
///         Self::new(numerator, denominator)
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
///     type Output = Self;
///
///     fn div(self, rhs: Scalar) -> Self::Output {
///         Self { value: self.value.iter().map(|v| v / rhs.value).collect() }
///     }
/// }
///
/// let scalar = Scalar { value: 2f32 };
/// let vector = Vector { value: vec![2f32, 4f32, 6f32] };
/// assert_eq!(vector / scalar, Vector { value: vec![1f32, 2f32, 3f32] });
/// ```
#[lang = "div"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    message = "cannot divide `{Self}` by `{Rhs}`",
    label = "no implementation for `{Self} / {Rhs}`"
)]
#[doc(alias = "/")]
pub trait Div<Rhs = Self> {
    /// The resulting type after applying the `/` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `/` operation.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(12 / 2, 6);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "div"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn div(self, rhs: Rhs) -> Self::Output;
}

macro_rules! div_impl_integer {
    ($(($($t:ty)*) => $panic:expr),*) => ($($(
        /// This operation rounds towards zero, truncating any
        /// fractional part of the exact result.
        ///
        /// # Panics
        ///
        #[doc = $panic]
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Div for $t {
            type Output = $t;

            #[inline]
            fn div(self, other: $t) -> $t { self / other }
        }

        forward_ref_binop! { impl Div, div for $t, $t }
    )*)*)
}

div_impl_integer! {
    (usize u8 u16 u32 u64 u128) => "This operation will panic if `other == 0`.",
    (isize i8 i16 i32 i64 i128) => "This operation will panic if `other == 0` or the division results in overflow."
}

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
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
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
/// struct SplitSlice<'a, T> {
///     slice: &'a [T],
/// }
///
/// impl<'a, T> Rem<usize> for SplitSlice<'a, T> {
///     type Output = Self;
///
///     fn rem(self, modulus: usize) -> Self::Output {
///         let len = self.slice.len();
///         let rem = len % modulus;
///         let start = len - rem;
///         Self {slice: &self.slice[start..]}
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
#[rustc_on_unimplemented(
    message = "cannot calculate the remainder of `{Self}` divided by `{Rhs}`",
    label = "no implementation for `{Self} % {Rhs}`"
)]
#[doc(alias = "%")]
pub trait Rem<Rhs = Self> {
    /// The resulting type after applying the `%` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `%` operation.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!(12 % 10, 2);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "rem"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rem(self, rhs: Rhs) -> Self::Output;
}

macro_rules! rem_impl_integer {
    ($(($($t:ty)*) => $panic:expr),*) => ($($(
        /// This operation satisfies `n % d == n - (n / d) * d`. The
        /// result has the same sign as the left operand.
        ///
        /// # Panics
        ///
        #[doc = $panic]
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Rem for $t {
            type Output = $t;

            #[inline]
            fn rem(self, other: $t) -> $t { self % other }
        }

        forward_ref_binop! { impl Rem, rem for $t, $t }
    )*)*)
}

rem_impl_integer! {
    (usize u8 u16 u32 u64 u128) => "This operation will panic if `other == 0`.",
    (isize i8 i16 i32 i64 i128) => "This operation will panic if `other == 0` or if `self / other` results in overflow."
}

macro_rules! rem_impl_float {
    ($($t:ty)*) => ($(

        /// The remainder from the division of two floats.
        ///
        /// The remainder has the same sign as the dividend and is computed as:
        /// `x - (x / y).trunc() * y`.
        ///
        /// # Examples
        /// ```
        /// let x: f32 = 50.50;
        /// let y: f32 = 8.125;
        /// let remainder = x - (x / y).trunc() * y;
        ///
        /// // The answer to both operations is 1.75
        /// assert_eq!(x % y, remainder);
        /// ```
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
///     type Output = Self;
///
///     fn neg(self) -> Self::Output {
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
    ///
    /// # Example
    ///
    /// ```
    /// let x: i32 = 12;
    /// assert_eq!(-x, -12);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[rustc_diagnostic_item = "neg"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn neg(self) -> Self::Output;
}

macro_rules! neg_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Neg for $t {
            type Output = $t;

            #[inline]
            #[rustc_inherit_overflow_checks]
            fn neg(self) -> $t { -self }
        }

        forward_ref_unop! { impl Neg, neg for $t }
    )*)
}

neg_impl! { isize i8 i16 i32 i64 i128 f32 f64 }

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
/// #[derive(Debug, Copy, Clone, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl AddAssign for Point {
///     fn add_assign(&mut self, other: Self) {
///         *self = Self {
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
#[rustc_on_unimplemented(
    message = "cannot add-assign `{Rhs}` to `{Self}`",
    label = "no implementation for `{Self} += {Rhs}`"
)]
#[doc(alias = "+")]
#[doc(alias = "+=")]
pub trait AddAssign<Rhs = Self> {
    /// Performs the `+=` operation.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x: u32 = 12;
    /// x += 1;
    /// assert_eq!(x, 13);
    /// ```
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

        forward_ref_op_assign! { impl AddAssign, add_assign for $t, $t }
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
/// #[derive(Debug, Copy, Clone, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32,
/// }
///
/// impl SubAssign for Point {
///     fn sub_assign(&mut self, other: Self) {
///         *self = Self {
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
#[rustc_on_unimplemented(
    message = "cannot subtract-assign `{Rhs}` from `{Self}`",
    label = "no implementation for `{Self} -= {Rhs}`"
)]
#[doc(alias = "-")]
#[doc(alias = "-=")]
pub trait SubAssign<Rhs = Self> {
    /// Performs the `-=` operation.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x: u32 = 12;
    /// x -= 1;
    /// assert_eq!(x, 11);
    /// ```
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

        forward_ref_op_assign! { impl SubAssign, sub_assign for $t, $t }
    )+)
}

sub_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

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
#[rustc_on_unimplemented(
    message = "cannot multiply-assign `{Self}` by `{Rhs}`",
    label = "no implementation for `{Self} *= {Rhs}`"
)]
#[doc(alias = "*")]
#[doc(alias = "*=")]
pub trait MulAssign<Rhs = Self> {
    /// Performs the `*=` operation.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x: u32 = 12;
    /// x *= 2;
    /// assert_eq!(x, 24);
    /// ```
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

        forward_ref_op_assign! { impl MulAssign, mul_assign for $t, $t }
    )+)
}

mul_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

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
#[rustc_on_unimplemented(
    message = "cannot divide-assign `{Self}` by `{Rhs}`",
    label = "no implementation for `{Self} /= {Rhs}`"
)]
#[doc(alias = "/")]
#[doc(alias = "/=")]
pub trait DivAssign<Rhs = Self> {
    /// Performs the `/=` operation.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x: u32 = 12;
    /// x /= 2;
    /// assert_eq!(x, 6);
    /// ```
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

        forward_ref_op_assign! { impl DivAssign, div_assign for $t, $t }
    )+)
}

div_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }

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
#[rustc_on_unimplemented(
    message = "cannot calculate and assign the remainder of `{Self}` divided by `{Rhs}`",
    label = "no implementation for `{Self} %= {Rhs}`"
)]
#[doc(alias = "%")]
#[doc(alias = "%=")]
pub trait RemAssign<Rhs = Self> {
    /// Performs the `%=` operation.
    ///
    /// # Example
    ///
    /// ```
    /// let mut x: u32 = 12;
    /// x %= 10;
    /// assert_eq!(x, 2);
    /// ```
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

        forward_ref_op_assign! { impl RemAssign, rem_assign for $t, $t }
    )+)
}

rem_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 }
