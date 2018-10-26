// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    /// The resulting type after applying the `!` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the unary `!` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn not(self) -> Self::Output;
}

macro_rules! not_impl {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Not for $t {
            type Output = $t;

            #[inline]
            fn not(self) -> $t { !self }
        }

        forward_ref_unop! { impl Not, not for $t }
    )*)
}

not_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

/// The bitwise AND operator `&`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
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
///     fn bitand(self, rhs: Self) -> Self {
///         Scalar(self.0 & rhs.0)
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
///     fn bitand(self, BooleanVector(rhs): Self) -> Self {
///         let BooleanVector(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         BooleanVector(lhs.iter().zip(rhs.iter()).map(|(x, y)| *x && *y).collect())
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
#[rustc_on_unimplemented(message="no implementation for `{Self} & {RHS}`",
                         label="no implementation for `{Self} & {RHS}`")]
pub trait BitAnd<RHS=Self> {
    /// The resulting type after applying the `&` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `&` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitand(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitand_impl {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitAnd for $t {
            type Output = $t;

            #[inline]
            fn bitand(self, rhs: $t) -> $t { self & rhs }
        }

        forward_ref_binop! { impl BitAnd, bitand for $t, $t }
    )*)
}

bitand_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

/// The bitwise OR operator `|`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
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
///     fn bitor(self, rhs: Self) -> Self {
///         Scalar(self.0 | rhs.0)
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
///     fn bitor(self, BooleanVector(rhs): Self) -> Self {
///         let BooleanVector(lhs) = self;
///         assert_eq!(lhs.len(), rhs.len());
///         BooleanVector(lhs.iter().zip(rhs.iter()).map(|(x, y)| *x || *y).collect())
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
#[rustc_on_unimplemented(message="no implementation for `{Self} | {RHS}`",
                         label="no implementation for `{Self} | {RHS}`")]
pub trait BitOr<RHS=Self> {
    /// The resulting type after applying the `|` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `|` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitor(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitor_impl {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for $t {
            type Output = $t;

            #[inline]
            fn bitor(self, rhs: $t) -> $t { self | rhs }
        }

        forward_ref_binop! { impl BitOr, bitor for $t, $t }
    )*)
}

bitor_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

/// The bitwise XOR operator `^`.
///
/// Note that `RHS` is `Self` by default, but this is not mandatory.
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
///     fn bitxor(self, rhs: Self) -> Self {
///         Scalar(self.0 ^ rhs.0)
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
/// let bv1 = BooleanVector(vec![true, true, false, false]);
/// let bv2 = BooleanVector(vec![true, false, true, false]);
/// let expected = BooleanVector(vec![false, true, true, false]);
/// assert_eq!(bv1 ^ bv2, expected);
/// ```
#[lang = "bitxor"]
#[doc(alias = "^")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="no implementation for `{Self} ^ {RHS}`",
                         label="no implementation for `{Self} ^ {RHS}`")]
pub trait BitXor<RHS=Self> {
    /// The resulting type after applying the `^` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `^` operation.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bitxor(self, rhs: RHS) -> Self::Output;
}

macro_rules! bitxor_impl {
    ($($t:ty),*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitXor for $t {
            type Output = $t;

            #[inline]
            fn bitxor(self, other: $t) -> $t { self ^ other }
        }

        forward_ref_binop! { impl BitXor, bitxor for $t, $t }
    )*)
}

bitxor_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

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
///     fn shl(self, Scalar(rhs): Self) -> Scalar {
///         let Scalar(lhs) = self;
///         Scalar(lhs << rhs)
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
///     fn shl(self, rhs: usize) -> SpinVector<T> {
///         // Rotate the vector by `rhs` places.
///         let (a, b) = self.vec.split_at(rhs);
///         let mut spun_vector: Vec<T> = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         SpinVector { vec: spun_vector }
///     }
/// }
///
/// assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } << 2,
///            SpinVector { vec: vec![2, 3, 4, 0, 1] });
/// ```
#[lang = "shl"]
#[doc(alias = "<<")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="no implementation for `{Self} << {RHS}`",
                         label="no implementation for `{Self} << {RHS}`")]
pub trait Shl<RHS=Self> {
    /// The resulting type after applying the `<<` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `<<` operation.
    #[must_use]
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
    ($($t:ty),*) => ($(
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

shl_impl_all! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, isize, i128 }

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
///     fn shr(self, Scalar(rhs): Self) -> Scalar {
///         let Scalar(lhs) = self;
///         Scalar(lhs >> rhs)
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
///     fn shr(self, rhs: usize) -> SpinVector<T> {
///         // Rotate the vector by `rhs` places.
///         let (a, b) = self.vec.split_at(self.vec.len() - rhs);
///         let mut spun_vector: Vec<T> = vec![];
///         spun_vector.extend_from_slice(b);
///         spun_vector.extend_from_slice(a);
///         SpinVector { vec: spun_vector }
///     }
/// }
///
/// assert_eq!(SpinVector { vec: vec![0, 1, 2, 3, 4] } >> 2,
///            SpinVector { vec: vec![3, 4, 0, 1, 2] });
/// ```
#[lang = "shr"]
#[doc(alias = ">>")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(message="no implementation for `{Self} >> {RHS}`",
                         label="no implementation for `{Self} >> {RHS}`")]
pub trait Shr<RHS=Self> {
    /// The resulting type after applying the `>>` operator.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Output;

    /// Performs the `>>` operation.
    #[must_use]
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
    ($($t:ty),*) => ($(
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

shr_impl_all! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }

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
///         *self = Scalar(self.0 & rhs.0)
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
///         *self = BooleanVector(self.0
///                                   .iter()
///                                   .zip(rhs.0.iter())
///                                   .map(|(x, y)| *x && *y)
///                                   .collect());
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
#[rustc_on_unimplemented(message="no implementation for `{Self} &= {Rhs}`",
                         label="no implementation for `{Self} &= {Rhs}`")]
pub trait BitAndAssign<Rhs=Self> {
    /// Performs the `&=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitand_assign(&mut self, rhs: Rhs);
}

macro_rules! bitand_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitAndAssign for $t {
            #[inline]
            fn bitand_assign(&mut self, other: $t) { *self &= other }
        }

        forward_ref_op_assign! { impl BitAndAssign, bitand_assign for $t, $t }
    )+)
}

bitand_assign_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

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
#[rustc_on_unimplemented(message="no implementation for `{Self} |= {Rhs}`",
                         label="no implementation for `{Self} |= {Rhs}`")]
pub trait BitOrAssign<Rhs=Self> {
    /// Performs the `|=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitor_assign(&mut self, rhs: Rhs);
}

macro_rules! bitor_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitOrAssign for $t {
            #[inline]
            fn bitor_assign(&mut self, other: $t) { *self |= other }
        }

        forward_ref_op_assign! { impl BitOrAssign, bitor_assign for $t, $t }
    )+)
}

bitor_assign_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

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
#[rustc_on_unimplemented(message="no implementation for `{Self} ^= {Rhs}`",
                         label="no implementation for `{Self} ^= {Rhs}`")]
pub trait BitXorAssign<Rhs=Self> {
    /// Performs the `^=` operation.
    #[stable(feature = "op_assign_traits", since = "1.8.0")]
    fn bitxor_assign(&mut self, rhs: Rhs);
}

macro_rules! bitxor_assign_impl {
    ($($t:ty),+) => ($(
        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitXorAssign for $t {
            #[inline]
            fn bitxor_assign(&mut self, other: $t) { *self ^= other }
        }

        forward_ref_op_assign! { impl BitXorAssign, bitxor_assign for $t, $t }
    )+)
}

bitxor_assign_impl! { bool, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128 }

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
#[rustc_on_unimplemented(message="no implementation for `{Self} <<= {Rhs}`",
                         label="no implementation for `{Self} <<= {Rhs}`")]
pub trait ShlAssign<Rhs=Self> {
    /// Performs the `<<=` operation.
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

        forward_ref_op_assign! { impl ShlAssign, shl_assign for $t, $f }
    )
}

macro_rules! shl_assign_impl_all {
    ($($t:ty),*) => ($(
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

shl_assign_impl_all! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }

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
#[rustc_on_unimplemented(message="no implementation for `{Self} >>= {Rhs}`",
                         label="no implementation for `{Self} >>= {Rhs}`")]
pub trait ShrAssign<Rhs=Self> {
    /// Performs the `>>=` operation.
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

        forward_ref_op_assign! { impl ShrAssign, shr_assign for $t, $f }
    )
}

macro_rules! shr_assign_impl_all {
    ($($t:ty),*) => ($(
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

shr_assign_impl_all! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }
