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
///     type Output = Self;
///
///     fn not(self) -> Self::Output {
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
#[doc(alias = "!")]
pub trait Not {
    /// The resulting type after applying the `!` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the unary `!` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(!true, false);
    /// assert_eq!(!false, true);
    /// assert_eq!(!1u8, 254);
    /// assert_eq!(!0u8, 255);
    /// ```
    #[must_use]
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

#[stable(feature = "not_never", since = "1.60.0")]
impl Not for ! {
    type Output = !;

    #[inline]
    fn not(self) -> ! {
        match self {}
    }
}

/// The bitwise AND operator `&`.
///
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
///
/// An implementation of `BitAnd` for a wrapper around `bool`.
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
///     fn bitand(self, rhs: Self) -> Self::Output {
///         Self(self.0 & rhs.0)
///     }
/// }
///
/// assert_eq!(Scalar(true) & Scalar(true), Scalar(true));
/// assert_eq!(Scalar(true) & Scalar(false), Scalar(false));
/// assert_eq!(Scalar(false) & Scalar(true), Scalar(false));
/// assert_eq!(Scalar(false) & Scalar(false), Scalar(false));
/// ```
///
/// An implementation of `BitAnd` for a wrapper around `Vec<bool>`.
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
///     fn bitand(self, Self(rhs): Self) -> Self::Output {
///         let Self(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         Self(
///             lhs.iter()
///                 .zip(rhs.iter())
///                 .map(|(x, y)| *x & *y)
///                 .collect()
///         )
///     }
/// }
///
/// let bv1 = BooleanVector(vec![true, true, false, false]);
/// let bv2 = BooleanVector(vec![true, false, true, false]);
/// let expected = BooleanVector(vec![true, false, false, false]);
/// assert_eq!(bv1 & bv2, expected);
/// ```
#[lang = "bitand"]
#[doc(alias = "&")]
#[stable(feature = "rust1", since = "1.0.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} & {Rhs}`",
    label = "no implementation for `{Self} & {Rhs}`"
)]
pub trait BitAnd<Rhs = Self> {
    /// The resulting type after applying the `&` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `&` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(true & false, false);
    /// assert_eq!(true & true, true);
    /// assert_eq!(5u8 & 1u8, 1);
    /// assert_eq!(5u8 & 2u8, 0);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitand(self, rhs: Rhs) -> Self::Output;
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
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
///
/// An implementation of `BitOr` for a wrapper around `bool`.
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
///     fn bitor(self, rhs: Self) -> Self::Output {
///         Self(self.0 | rhs.0)
///     }
/// }
///
/// assert_eq!(Scalar(true) | Scalar(true), Scalar(true));
/// assert_eq!(Scalar(true) | Scalar(false), Scalar(true));
/// assert_eq!(Scalar(false) | Scalar(true), Scalar(true));
/// assert_eq!(Scalar(false) | Scalar(false), Scalar(false));
/// ```
///
/// An implementation of `BitOr` for a wrapper around `Vec<bool>`.
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
///     fn bitor(self, Self(rhs): Self) -> Self::Output {
///         let Self(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         Self(
///             lhs.iter()
///                 .zip(rhs.iter())
///                 .map(|(x, y)| *x | *y)
///                 .collect()
///         )
///     }
/// }
///
/// let bv1 = BooleanVector(vec![true, true, false, false]);
/// let bv2 = BooleanVector(vec![true, false, true, false]);
/// let expected = BooleanVector(vec![true, true, true, false]);
/// assert_eq!(bv1 | bv2, expected);
/// ```
#[lang = "bitor"]
#[doc(alias = "|")]
#[stable(feature = "rust1", since = "1.0.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} | {Rhs}`",
    label = "no implementation for `{Self} | {Rhs}`"
)]
pub trait BitOr<Rhs = Self> {
    /// The resulting type after applying the `|` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `|` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(true | false, true);
    /// assert_eq!(false | false, false);
    /// assert_eq!(5u8 | 1u8, 5);
    /// assert_eq!(5u8 | 2u8, 7);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitor(self, rhs: Rhs) -> Self::Output;
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
/// Note that `Rhs` is `Self` by default, but this is not mandatory.
///
/// # Examples
///
/// An implementation of `BitXor` that lifts `^` to a wrapper around `bool`.
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
///     fn bitxor(self, rhs: Self) -> Self::Output {
///         Self(self.0 ^ rhs.0)
///     }
/// }
///
/// assert_eq!(Scalar(true) ^ Scalar(true), Scalar(false));
/// assert_eq!(Scalar(true) ^ Scalar(false), Scalar(true));
/// assert_eq!(Scalar(false) ^ Scalar(true), Scalar(true));
/// assert_eq!(Scalar(false) ^ Scalar(false), Scalar(false));
/// ```
///
/// An implementation of `BitXor` trait for a wrapper around `Vec<bool>`.
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
///     fn bitxor(self, Self(rhs): Self) -> Self::Output {
///         let Self(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         Self(
///             lhs.iter()
///                 .zip(rhs.iter())
///                 .map(|(x, y)| *x ^ *y)
///                 .collect()
///         )
///     }
/// }
///
/// let bv1 = BooleanVector(vec![true, true, false, false]);
/// let bv2 = BooleanVector(vec![true, false, true, false]);
/// let expected = BooleanVector(vec![false, true, true, false]);
/// assert_eq!(bv1 ^ bv2, expected);
/// ```
#[lang = "bitxor"]
#[doc(alias = "^")]
#[stable(feature = "rust1", since = "1.0.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} ^ {Rhs}`",
    label = "no implementation for `{Self} ^ {Rhs}`"
)]
pub trait BitXor<Rhs = Self> {
    /// The resulting type after applying the `^` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `^` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(true ^ false, true);
    /// assert_eq!(true ^ true, false);
    /// assert_eq!(5u8 ^ 1u8, 4);
    /// assert_eq!(5u8 ^ 2u8, 7);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitxor(self, rhs: Rhs) -> Self::Output;
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

/// The left shift operator `<<`. Note that because this trait is implemented
/// for all integer types with multiple right-hand-side types, Rust's type
/// checker has special handling for `_ << _`, setting the result type for
/// integer operations to the type of the left-hand-side operand. This means
/// that though `a << b` and `a.shl(b)` are one and the same from an evaluation
/// standpoint, they are different when it comes to type inference.
///
/// # Examples
///
/// An implementation of `Shl` that lifts the `<<` operation on integers to a
/// wrapper around `usize`.
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
///     fn shl(self, Self(rhs): Self) -> Self::Output {
///         let Self(lhs) = self;
///         Self(lhs << rhs)
///     }
/// }
///
/// assert_eq!(Scalar(4) << Scalar(2), Scalar(16));
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
///     fn shl(self, rhs: usize) -> Self::Output {
///         // Rotate the vector by `rhs` places.
///         let (a, b) = self.vec.split_at(rhs);
///         let mut spun_vector = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         Self { vec: spun_vector }
///     }
/// }
///
/// assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } << 2,
///            SpinVector { vec: vec![2, 3, 4, 0, 1] });
/// ```
#[lang = "shl"]
#[doc(alias = "<<")]
#[stable(feature = "rust1", since = "1.0.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} << {Rhs}`",
    label = "no implementation for `{Self} << {Rhs}`"
)]
pub trait Shl<Rhs = Self> {
    /// The resulting type after applying the `<<` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `<<` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(5u8 << 1, 10);
    /// assert_eq!(1u8 << 1, 2);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn shl(self, rhs: Rhs) -> Self::Output;
}

macro_rules! shl_impl {
    ($t:ty, $f:ty) => {
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
    };
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

shl_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

/// The right shift operator `>>`. Note that because this trait is implemented
/// for all integer types with multiple right-hand-side types, Rust's type
/// checker has special handling for `_ >> _`, setting the result type for
/// integer operations to the type of the left-hand-side operand. This means
/// that though `a >> b` and `a.shr(b)` are one and the same from an evaluation
/// standpoint, they are different when it comes to type inference.
///
/// # Examples
///
/// An implementation of `Shr` that lifts the `>>` operation on integers to a
/// wrapper around `usize`.
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
///     fn shr(self, Self(rhs): Self) -> Self::Output {
///         let Self(lhs) = self;
///         Self(lhs >> rhs)
///     }
/// }
///
/// assert_eq!(Scalar(16) >> Scalar(2), Scalar(4));
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
///     fn shr(self, rhs: usize) -> Self::Output {
///         // Rotate the vector by `rhs` places.
///         let (a, b) = self.vec.split_at(self.vec.len() - rhs);
///         let mut spun_vector = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         Self { vec: spun_vector }
///     }
/// }
///
/// assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } >> 2,
///            SpinVector { vec: vec![3, 4, 0, 1, 2] });
/// ```
#[lang = "shr"]
#[doc(alias = ">>")]
#[stable(feature = "rust1", since = "1.0.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} >> {Rhs}`",
    label = "no implementation for `{Self} >> {Rhs}`"
)]
pub trait Shr<Rhs = Self> {
    /// The resulting type after applying the `>>` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `>>` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(5u8 >> 1, 2);
    /// assert_eq!(2u8 >> 1, 1);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn shr(self, rhs: Rhs) -> Self::Output;
}

macro_rules! shr_impl {
    ($t:ty, $f:ty) => {
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
    };
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

/// The bitwise AND assignment operator `&=`.
///
/// # Examples
///
/// An implementation of `BitAndAssign` that lifts the `&=` operator to a
/// wrapper around `bool`.
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
///         *self = Self(self.0 & rhs.0)
///     }
/// }
///
/// let mut scalar = Scalar(true);
/// scalar &= Scalar(true);
/// assert_eq!(scalar, Scalar(true));
///
/// let mut scalar = Scalar(true);
/// scalar &= Scalar(false);
/// assert_eq!(scalar, Scalar(false));
///
/// let mut scalar = Scalar(false);
/// scalar &= Scalar(true);
/// assert_eq!(scalar, Scalar(false));
///
/// let mut scalar = Scalar(false);
/// scalar &= Scalar(false);
/// assert_eq!(scalar, Scalar(false));
/// ```
///
/// Here, the `BitAndAssign` trait is implemented for a wrapper around
/// `Vec<bool>`.
///
/// ```
/// use std::ops::BitAndAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct BooleanVector(Vec<bool>);
///
/// impl BitAndAssign for BooleanVector {
///     // `rhs` is the "right-hand side" of the expression `a &= b`.
///     fn bitand_assign(&mut self, rhs: Self) {
///         assert_eq!(self.0.len(), rhs.0.len());
///         *self = Self(
///             self.0
///                 .iter()
///                 .zip(rhs.0.iter())
///                 .map(|(x, y)| *x & *y)
///                 .collect()
///         );
///     }
/// }
///
/// let mut bv = BooleanVector(vec![true, true, false, false]);
/// bv &= BooleanVector(vec![true, false, true, false]);
/// let expected = BooleanVector(vec![true, false, false, false]);
/// assert_eq!(bv, expected);
/// ```
#[lang = "bitand_assign"]
#[doc(alias = "&=")]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} &= {Rhs}`",
    label = "no implementation for `{Self} &= {Rhs}`"
)]
pub trait BitAndAssign<Rhs = Self> {
    /// Performs the `&=` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = true;
    /// x &= false;
    /// assert_eq!(x, false);
    ///
    /// let mut x = true;
    /// x &= true;
    /// assert_eq!(x, true);
    ///
    /// let mut x: u8 = 5;
    /// x &= 1;
    /// assert_eq!(x, 1);
    ///
    /// let mut x: u8 = 5;
    /// x &= 2;
    /// assert_eq!(x, 0);
    /// ```
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

        forward_ref_op_assign! { impl BitAndAssign, bitand_assign for $t, $t }
    )+)
}

bitand_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise OR assignment operator `|=`.
///
/// # Examples
///
/// ```
/// use std::ops::BitOrAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct PersonalPreferences {
///     likes_cats: bool,
///     likes_dogs: bool,
/// }
///
/// impl BitOrAssign for PersonalPreferences {
///     fn bitor_assign(&mut self, rhs: Self) {
///         self.likes_cats |= rhs.likes_cats;
///         self.likes_dogs |= rhs.likes_dogs;
///     }
/// }
///
/// let mut prefs = PersonalPreferences { likes_cats: true, likes_dogs: false };
/// prefs |= PersonalPreferences { likes_cats: false, likes_dogs: true };
/// assert_eq!(prefs, PersonalPreferences { likes_cats: true, likes_dogs: true });
/// ```
#[lang = "bitor_assign"]
#[doc(alias = "|=")]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} |= {Rhs}`",
    label = "no implementation for `{Self} |= {Rhs}`"
)]
pub trait BitOrAssign<Rhs = Self> {
    /// Performs the `|=` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = true;
    /// x |= false;
    /// assert_eq!(x, true);
    ///
    /// let mut x = false;
    /// x |= false;
    /// assert_eq!(x, false);
    ///
    /// let mut x: u8 = 5;
    /// x |= 1;
    /// assert_eq!(x, 5);
    ///
    /// let mut x: u8 = 5;
    /// x |= 2;
    /// assert_eq!(x, 7);
    /// ```
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

        forward_ref_op_assign! { impl BitOrAssign, bitor_assign for $t, $t }
    )+)
}

bitor_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The bitwise XOR assignment operator `^=`.
///
/// # Examples
///
/// ```
/// use std::ops::BitXorAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Personality {
///     has_soul: bool,
///     likes_knitting: bool,
/// }
///
/// impl BitXorAssign for Personality {
///     fn bitxor_assign(&mut self, rhs: Self) {
///         self.has_soul ^= rhs.has_soul;
///         self.likes_knitting ^= rhs.likes_knitting;
///     }
/// }
///
/// let mut personality = Personality { has_soul: false, likes_knitting: true };
/// personality ^= Personality { has_soul: true, likes_knitting: true };
/// assert_eq!(personality, Personality { has_soul: true, likes_knitting: false});
/// ```
#[lang = "bitxor_assign"]
#[doc(alias = "^=")]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} ^= {Rhs}`",
    label = "no implementation for `{Self} ^= {Rhs}`"
)]
pub trait BitXorAssign<Rhs = Self> {
    /// Performs the `^=` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = true;
    /// x ^= false;
    /// assert_eq!(x, true);
    ///
    /// let mut x = true;
    /// x ^= true;
    /// assert_eq!(x, false);
    ///
    /// let mut x: u8 = 5;
    /// x ^= 1;
    /// assert_eq!(x, 4);
    ///
    /// let mut x: u8 = 5;
    /// x ^= 2;
    /// assert_eq!(x, 7);
    /// ```
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

        forward_ref_op_assign! { impl BitXorAssign, bitxor_assign for $t, $t }
    )+)
}

bitxor_assign_impl! { bool usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The left shift assignment operator `<<=`.
///
/// # Examples
///
/// An implementation of `ShlAssign` for a wrapper around `usize`.
///
/// ```
/// use std::ops::ShlAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(usize);
///
/// impl ShlAssign<usize> for Scalar {
///     fn shl_assign(&mut self, rhs: usize) {
///         self.0 <<= rhs;
///     }
/// }
///
/// let mut scalar = Scalar(4);
/// scalar <<= 2;
/// assert_eq!(scalar, Scalar(16));
/// ```
#[lang = "shl_assign"]
#[doc(alias = "<<=")]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} <<= {Rhs}`",
    label = "no implementation for `{Self} <<= {Rhs}`"
)]
pub trait ShlAssign<Rhs = Self> {
    /// Performs the `<<=` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x: u8 = 5;
    /// x <<= 1;
    /// assert_eq!(x, 10);
    ///
    /// let mut x: u8 = 1;
    /// x <<= 1;
    /// assert_eq!(x, 2);
    /// ```
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn shl_assign(&mut self, rhs: Rhs);
}

macro_rules! shl_assign_impl {
    ($t:ty, $f:ty) => {
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShlAssign<$f> for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn shl_assign(&mut self, other: $f) {
                *self <<= other
            }
        }

        forward_ref_op_assign! { impl ShlAssign, shl_assign for $t, $f }
    };
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
/// An implementation of `ShrAssign` for a wrapper around `usize`.
///
/// ```
/// use std::ops::ShrAssign;
///
/// #[derive(Debug, PartialEq)]
/// struct Scalar(usize);
///
/// impl ShrAssign<usize> for Scalar {
///     fn shr_assign(&mut self, rhs: usize) {
///         self.0 >>= rhs;
///     }
/// }
///
/// let mut scalar = Scalar(16);
/// scalar >>= 2;
/// assert_eq!(scalar, Scalar(4));
/// ```
#[lang = "shr_assign"]
#[doc(alias = ">>=")]
#[stable(feature = "op_assign_traits", since = "1.8.0")]
#[diagnostic::on_unimplemented(
    message = "no implementation for `{Self} >>= {Rhs}`",
    label = "no implementation for `{Self} >>= {Rhs}`"
)]
pub trait ShrAssign<Rhs = Self> {
    /// Performs the `>>=` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x: u8 = 5;
    /// x >>= 1;
    /// assert_eq!(x, 2);
    ///
    /// let mut x: u8 = 2;
    /// x >>= 1;
    /// assert_eq!(x, 1);
    /// ```
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn shr_assign(&mut self, rhs: Rhs);
}

macro_rules! shr_assign_impl {
    ($t:ty, $f:ty) => {
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for $t {
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn shr_assign(&mut self, other: $f) {
                *self >>= other
            }
        }

        forward_ref_op_assign! { impl ShrAssign, shr_assign for $t, $f }
    };
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
