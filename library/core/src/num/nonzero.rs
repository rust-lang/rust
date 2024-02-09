//! Definitions of integer that is known not to equal zero.

use crate::cmp::Ordering;
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::intrinsics;
#[cfg(bootstrap)]
use crate::marker::StructuralEq;
use crate::marker::StructuralPartialEq;
use crate::ops::{BitOr, BitOrAssign, Div, Neg, Rem};
use crate::str::FromStr;

use super::from_str_radix;
use super::{IntErrorKind, ParseIntError};

mod private {
    #[unstable(
        feature = "nonzero_internals",
        reason = "implementation detail which may disappear or be replaced at any time",
        issue = "none"
    )]
    #[const_trait]
    pub trait Sealed {}
}

/// A marker trait for primitive types which can be zero.
///
/// This is an implementation detail for [`NonZero<T>`](NonZero) which may disappear or be replaced at any time.
#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[const_trait]
pub trait ZeroablePrimitive: Sized + Copy + private::Sealed {}

macro_rules! impl_zeroable_primitive {
    ($primitive:ty) => {
        #[unstable(
            feature = "nonzero_internals",
            reason = "implementation detail which may disappear or be replaced at any time",
            issue = "none"
        )]
        impl const private::Sealed for $primitive {}

        #[unstable(
            feature = "nonzero_internals",
            reason = "implementation detail which may disappear or be replaced at any time",
            issue = "none"
        )]
        impl const ZeroablePrimitive for $primitive {}
    };
}

impl_zeroable_primitive!(u8);
impl_zeroable_primitive!(u16);
impl_zeroable_primitive!(u32);
impl_zeroable_primitive!(u64);
impl_zeroable_primitive!(u128);
impl_zeroable_primitive!(usize);
impl_zeroable_primitive!(i8);
impl_zeroable_primitive!(i16);
impl_zeroable_primitive!(i32);
impl_zeroable_primitive!(i64);
impl_zeroable_primitive!(i128);
impl_zeroable_primitive!(isize);

/// A value that is known not to equal zero.
///
/// This enables some memory layout optimization.
/// For example, `Option<NonZero<u32>>` is the same size as `u32`:
///
/// ```
/// #![feature(generic_nonzero)]
/// use core::mem::size_of;
///
/// assert_eq!(size_of::<Option<core::num::NonZero<u32>>>(), size_of::<u32>());
/// ```
#[unstable(feature = "generic_nonzero", issue = "120257")]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
#[rustc_diagnostic_item = "NonZero"]
pub struct NonZero<T: ZeroablePrimitive>(T);

impl<T> NonZero<T>
where
    T: ZeroablePrimitive,
{
    /// Creates a non-zero if the given value is not zero.
    #[stable(feature = "nonzero", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_nonzero_int_methods", since = "1.47.0")]
    #[must_use]
    #[inline]
    pub const fn new(n: T) -> Option<Self> {
        // SAFETY: Memory layout optimization guarantees that `Option<NonZero<T>>` has
        //         the same layout and size as `T`, with `0` representing `None`.
        unsafe { intrinsics::transmute_unchecked(n) }
    }

    /// Creates a non-zero without checking whether the value is non-zero.
    /// This results in undefined behaviour if the value is zero.
    ///
    /// # Safety
    ///
    /// The value must not be zero.
    #[stable(feature = "nonzero", since = "1.28.0")]
    #[rustc_const_stable(feature = "nonzero", since = "1.28.0")]
    #[must_use]
    #[inline]
    pub const unsafe fn new_unchecked(n: T) -> Self {
        match Self::new(n) {
            Some(n) => n,
            None => {
                // SAFETY: The caller guarantees that `n` is non-zero, so this is unreachable.
                unsafe {
                    intrinsics::assert_unsafe_precondition!(
                      "NonZero::new_unchecked requires the argument to be non-zero",
                      () => false,
                    );
                    intrinsics::unreachable()
                }
            }
        }
    }

    /// Converts a reference to a non-zero mutable reference
    /// if the referenced value is not zero.
    #[unstable(feature = "nonzero_from_mut", issue = "106290")]
    #[must_use]
    #[inline]
    pub fn from_mut(n: &mut T) -> Option<&mut Self> {
        // SAFETY: Memory layout optimization guarantees that `Option<NonZero<T>>` has
        //         the same layout and size as `T`, with `0` representing `None`.
        let opt_n = unsafe { &mut *(n as *mut T as *mut Option<Self>) };

        opt_n.as_mut()
    }

    /// Converts a mutable reference to a non-zero mutable reference
    /// without checking whether the referenced value is non-zero.
    /// This results in undefined behavior if the referenced value is zero.
    ///
    /// # Safety
    ///
    /// The referenced value must not be zero.
    #[unstable(feature = "nonzero_from_mut", issue = "106290")]
    #[must_use]
    #[inline]
    pub unsafe fn from_mut_unchecked(n: &mut T) -> &mut Self {
        match Self::from_mut(n) {
            Some(n) => n,
            None => {
                // SAFETY: The caller guarantees that `n` references a value that is non-zero, so this is unreachable.
                unsafe {
                    intrinsics::assert_unsafe_precondition!(
                      "NonZero::from_mut_unchecked requires the argument to dereference as non-zero",
                      () => false,
                    );
                    intrinsics::unreachable()
                }
            }
        }
    }
}

macro_rules! impl_nonzero_fmt {
    ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => {
        $(
            #[$stability]
            impl fmt::$Trait for $Ty {
                #[inline]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.get().fmt(f)
                }
            }
        )+
    }
}

macro_rules! nonzero_integer {
    (
        #[$stability:meta]
        Self = $Ty:ident,
        Primitive = $signedness:ident $Int:ident,
        $(UnsignedNonZero = $UnsignedNonZero:ident,)?
        UnsignedPrimitive = $UnsignedPrimitive:ty,

        // Used in doc comments.
        leading_zeros_test = $leading_zeros_test:expr,
    ) => {
        /// An integer that is known not to equal zero.
        ///
        /// This enables some memory layout optimization.
        #[doc = concat!("For example, `Option<", stringify!($Ty), ">` is the same size as `", stringify!($Int), "`:")]
        ///
        /// ```rust
        /// use std::mem::size_of;
        #[doc = concat!("assert_eq!(size_of::<Option<core::num::", stringify!($Ty), ">>(), size_of::<", stringify!($Int), ">());")]
        /// ```
        ///
        /// # Layout
        ///
        #[doc = concat!("`", stringify!($Ty), "` is guaranteed to have the same layout and bit validity as `", stringify!($Int), "`")]
        /// with the exception that `0` is not a valid instance.
        #[doc = concat!("`Option<", stringify!($Ty), ">` is guaranteed to be compatible with `", stringify!($Int), "`,")]
        /// including in FFI.
        ///
        /// Thanks to the [null pointer optimization],
        #[doc = concat!("`", stringify!($Ty), "` and `Option<", stringify!($Ty), ">`")]
        /// are guaranteed to have the same size and alignment:
        ///
        /// ```
        /// # use std::mem::{size_of, align_of};
        #[doc = concat!("use std::num::", stringify!($Ty), ";")]
        ///
        #[doc = concat!("assert_eq!(size_of::<", stringify!($Ty), ">(), size_of::<Option<", stringify!($Ty), ">>());")]
        #[doc = concat!("assert_eq!(align_of::<", stringify!($Ty), ">(), align_of::<Option<", stringify!($Ty), ">>());")]
        /// ```
        ///
        /// [null pointer optimization]: crate::option#representation
        #[$stability]
        pub type $Ty = NonZero<$Int>;

        impl $Ty {
            /// Returns the value as a primitive type.
            #[$stability]
            #[inline]
            #[rustc_const_stable(feature = "const_nonzero_get", since = "1.34.0")]
            pub const fn get(self) -> $Int {
                // FIXME: Remove this after LLVM supports `!range` metadata for function
                // arguments https://github.com/llvm/llvm-project/issues/76628
                //
                // Rustc can set range metadata only if it loads `self` from
                // memory somewhere. If the value of `self` was from by-value argument
                // of some not-inlined function, LLVM don't have range metadata
                // to understand that the value cannot be zero.

                // SAFETY: It is an invariant of this type.
                unsafe {
                    intrinsics::assume(self.0 != 0);
                }
                self.0
            }

            /// The size of this non-zero integer type in bits.
            ///
            #[doc = concat!("This value is equal to [`", stringify!($Int), "::BITS`].")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Ty), "::BITS, ", stringify!($Int), "::BITS);")]
            /// ```
            #[stable(feature = "nonzero_bits", since = "1.67.0")]
            pub const BITS: u32 = <$Int>::BITS;

            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// On many architectures, this function can perform better than `leading_zeros()` on the underlying integer type, as special handling of zero can be avoided.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("let n = std::num::", stringify!($Ty), "::new(", $leading_zeros_test, ").unwrap();")]
            ///
            /// assert_eq!(n.leading_zeros(), 0);
            /// ```
            #[stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[rustc_const_stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn leading_zeros(self) -> u32 {
                // SAFETY: since `self` cannot be zero, it is safe to call `ctlz_nonzero`.
                unsafe { intrinsics::ctlz_nonzero(self.get() as $UnsignedPrimitive) as u32 }
            }

            /// Returns the number of trailing zeros in the binary representation
            /// of `self`.
            ///
            /// On many architectures, this function can perform better than `trailing_zeros()` on the underlying integer type, as special handling of zero can be avoided.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("let n = std::num::", stringify!($Ty), "::new(0b0101000).unwrap();")]
            ///
            /// assert_eq!(n.trailing_zeros(), 3);
            /// ```
            #[stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[rustc_const_stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn trailing_zeros(self) -> u32 {
                // SAFETY: since `self` cannot be zero, it is safe to call `cttz_nonzero`.
                unsafe { intrinsics::cttz_nonzero(self.get() as $UnsignedPrimitive) as u32 }
            }

            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(non_zero_count_ones)]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("# use std::num::{self, ", stringify!($Ty), "};")]
            ///
            /// let one = num::NonZeroU32::new(1)?;
            /// let three = num::NonZeroU32::new(3)?;
            #[doc = concat!("let a = ", stringify!($Ty), "::new(0b100_0000)?;")]
            #[doc = concat!("let b = ", stringify!($Ty), "::new(0b100_0011)?;")]
            ///
            /// assert_eq!(a.count_ones(), one);
            /// assert_eq!(b.count_ones(), three);
            /// # Some(())
            /// # }
            /// ```
            ///
            #[unstable(feature = "non_zero_count_ones", issue = "120287")]
            #[rustc_const_unstable(feature = "non_zero_count_ones", issue = "120287")]
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn count_ones(self) -> NonZeroU32 {
                // SAFETY:
                // `self` is non-zero, which means it has at least one bit set, which means
                // that the result of `count_ones` is non-zero.
                unsafe { NonZeroU32::new_unchecked(self.get().count_ones()) }
            }

            nonzero_integer_signedness_dependent_methods! {
                Self = $Ty,
                Primitive = $signedness $Int,
                $(UnsignedNonZero = $UnsignedNonZero,)?
                UnsignedPrimitive = $UnsignedPrimitive,
            }

            /// Multiplies two non-zero integers together.
            /// Checks for overflow and returns [`None`] on overflow.
            /// As a consequence, the result cannot wrap to zero.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
            #[doc = concat!("let four = ", stringify!($Ty), "::new(4)?;")]
            #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                            stringify!($Int), "::MAX)?;")]
            ///
            /// assert_eq!(Some(four), two.checked_mul(two));
            /// assert_eq!(None, max.checked_mul(two));
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
            #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_mul(self, other: Self) -> Option<Self> {
                if let Some(result) = self.get().checked_mul(other.get()) {
                    // SAFETY:
                    // - `checked_mul` returns `None` on overflow
                    // - `self` and `other` are non-zero
                    // - the only way to get zero from a multiplication without overflow is for one
                    //   of the sides to be zero
                    //
                    // So the result cannot be zero.
                    Some(unsafe { Self::new_unchecked(result) })
                } else {
                    None
                }
            }

            /// Multiplies two non-zero integers together.
            #[doc = concat!("Return [`", stringify!($Ty), "::MAX`] on overflow.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
            #[doc = concat!("let four = ", stringify!($Ty), "::new(4)?;")]
            #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                            stringify!($Int), "::MAX)?;")]
            ///
            /// assert_eq!(four, two.saturating_mul(two));
            /// assert_eq!(max, four.saturating_mul(max));
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
            #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn saturating_mul(self, other: Self) -> Self {
                // SAFETY:
                // - `saturating_mul` returns `u*::MAX`/`i*::MAX`/`i*::MIN` on overflow/underflow,
                //   all of which are non-zero
                // - `self` and `other` are non-zero
                // - the only way to get zero from a multiplication without overflow is for one
                //   of the sides to be zero
                //
                // So the result cannot be zero.
                unsafe { Self::new_unchecked(self.get().saturating_mul(other.get())) }
            }

            /// Multiplies two non-zero integers together,
            /// assuming overflow cannot occur.
            /// Overflow is unchecked, and it is undefined behaviour to overflow
            /// *even if the result would wrap to a non-zero value*.
            /// The behaviour is undefined as soon as
            #[doc = sign_dependent_expr!{
                $signedness ?
                if signed {
                    concat!("`self * rhs > ", stringify!($Int), "::MAX`, ",
                            "or `self * rhs < ", stringify!($Int), "::MIN`.")
                }
                if unsigned {
                    concat!("`self * rhs > ", stringify!($Int), "::MAX`.")
                }
            }]
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(nonzero_ops)]
            ///
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
            #[doc = concat!("let four = ", stringify!($Ty), "::new(4)?;")]
            ///
            /// assert_eq!(four, unsafe { two.unchecked_mul(two) });
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_ops", issue = "84186")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const unsafe fn unchecked_mul(self, other: Self) -> Self {
                // SAFETY: The caller ensures there is no overflow.
                unsafe { Self::new_unchecked(self.get().unchecked_mul(other.get())) }
            }

            /// Raises non-zero value to an integer power.
            /// Checks for overflow and returns [`None`] on overflow.
            /// As a consequence, the result cannot wrap to zero.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let three = ", stringify!($Ty), "::new(3)?;")]
            #[doc = concat!("let twenty_seven = ", stringify!($Ty), "::new(27)?;")]
            #[doc = concat!("let half_max = ", stringify!($Ty), "::new(",
                            stringify!($Int), "::MAX / 2)?;")]
            ///
            /// assert_eq!(Some(twenty_seven), three.checked_pow(3));
            /// assert_eq!(None, half_max.checked_pow(3));
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
            #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_pow(self, other: u32) -> Option<Self> {
                if let Some(result) = self.get().checked_pow(other) {
                    // SAFETY:
                    // - `checked_pow` returns `None` on overflow/underflow
                    // - `self` is non-zero
                    // - the only way to get zero from an exponentiation without overflow is
                    //   for base to be zero
                    //
                    // So the result cannot be zero.
                    Some(unsafe { Self::new_unchecked(result) })
                } else {
                    None
                }
            }

            /// Raise non-zero value to an integer power.
            #[doc = sign_dependent_expr!{
                $signedness ?
                if signed {
                    concat!("Return [`", stringify!($Ty), "::MIN`] ",
                                "or [`", stringify!($Ty), "::MAX`] on overflow.")
                }
                if unsigned {
                    concat!("Return [`", stringify!($Ty), "::MAX`] on overflow.")
                }
            }]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let three = ", stringify!($Ty), "::new(3)?;")]
            #[doc = concat!("let twenty_seven = ", stringify!($Ty), "::new(27)?;")]
            #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                            stringify!($Int), "::MAX)?;")]
            ///
            /// assert_eq!(twenty_seven, three.saturating_pow(3));
            /// assert_eq!(max, max.saturating_pow(3));
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
            #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn saturating_pow(self, other: u32) -> Self {
                // SAFETY:
                // - `saturating_pow` returns `u*::MAX`/`i*::MAX`/`i*::MIN` on overflow/underflow,
                //   all of which are non-zero
                // - `self` is non-zero
                // - the only way to get zero from an exponentiation without overflow is
                //   for base to be zero
                //
                // So the result cannot be zero.
                unsafe { Self::new_unchecked(self.get().saturating_pow(other)) }
            }
        }

        #[$stability]
        impl Clone for $Ty {
            #[inline]
            fn clone(&self) -> Self {
                // SAFETY: The contained value is non-zero.
                unsafe { Self(self.0) }
            }
        }

        #[$stability]
        impl Copy for $Ty {}

        #[$stability]
        impl PartialEq for $Ty {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }

            #[inline]
            fn ne(&self, other: &Self) -> bool {
                self.0 != other.0
            }
        }

        #[unstable(feature = "structural_match", issue = "31434")]
        impl StructuralPartialEq for $Ty {}

        #[$stability]
        impl Eq for $Ty {}

        #[unstable(feature = "structural_match", issue = "31434")]
        #[cfg(bootstrap)]
        impl StructuralEq for $Ty {}

        #[$stability]
        impl PartialOrd for $Ty {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }

            #[inline]
            fn lt(&self, other: &Self) -> bool {
                self.0 < other.0
            }

            #[inline]
            fn le(&self, other: &Self) -> bool {
                self.0 <= other.0
            }

            #[inline]
            fn gt(&self, other: &Self) -> bool {
                self.0 > other.0
            }

            #[inline]
            fn ge(&self, other: &Self) -> bool {
                self.0 >= other.0
            }
        }

        #[$stability]
        impl Ord for $Ty {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.cmp(&other.0)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                // SAFETY: The maximum of two non-zero values is still non-zero.
                unsafe { Self(self.0.max(other.0)) }
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                // SAFETY: The minimum of two non-zero values is still non-zero.
                unsafe { Self(self.0.min(other.0)) }
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                // SAFETY: A non-zero value clamped between two non-zero values is still non-zero.
                unsafe { Self(self.0.clamp(min.0, max.0)) }
            }
        }

        #[$stability]
        impl Hash for $Ty {
            #[inline]
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher,
            {
                self.0.hash(state)
            }
        }

        #[stable(feature = "from_nonzero", since = "1.31.0")]
        impl From<$Ty> for $Int {
            #[doc = concat!("Converts a `", stringify!($Ty), "` into an `", stringify!($Int), "`")]
            #[inline]
            fn from(nonzero: $Ty) -> Self {
                // Call nonzero to keep information range information
                // from get method.
                nonzero.get()
            }
        }

        #[stable(feature = "nonzero_bitor", since = "1.45.0")]
        impl BitOr for $Ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                // SAFETY: since `self` and `rhs` are both nonzero, the
                // result of the bitwise-or will be nonzero.
                unsafe { Self::new_unchecked(self.get() | rhs.get()) }
            }
        }

        #[stable(feature = "nonzero_bitor", since = "1.45.0")]
        impl BitOr<$Int> for $Ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: $Int) -> Self::Output {
                // SAFETY: since `self` is nonzero, the result of the
                // bitwise-or will be nonzero regardless of the value of
                // `rhs`.
                unsafe { Self::new_unchecked(self.get() | rhs) }
            }
        }

        #[stable(feature = "nonzero_bitor", since = "1.45.0")]
        impl BitOr<$Ty> for $Int {
            type Output = $Ty;

            #[inline]
            fn bitor(self, rhs: $Ty) -> Self::Output {
                // SAFETY: since `rhs` is nonzero, the result of the
                // bitwise-or will be nonzero regardless of the value of
                // `self`.
                unsafe { $Ty::new_unchecked(self | rhs.get()) }
            }
        }

        #[stable(feature = "nonzero_bitor", since = "1.45.0")]
        impl BitOrAssign for $Ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }

        #[stable(feature = "nonzero_bitor", since = "1.45.0")]
        impl BitOrAssign<$Int> for $Ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: $Int) {
                *self = *self | rhs;
            }
        }

        impl_nonzero_fmt! {
            #[$stability] (Debug, Display, Binary, Octal, LowerHex, UpperHex) for $Ty
        }

        #[stable(feature = "nonzero_parse", since = "1.35.0")]
        impl FromStr for $Ty {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, Self::Err> {
                Self::new(from_str_radix(src, 10)?)
                    .ok_or(ParseIntError {
                        kind: IntErrorKind::Zero
                    })
            }
        }

        nonzero_integer_signedness_dependent_impls!($Ty $signedness $Int);
    };

    (Self = $Ty:ident, Primitive = unsigned $Int:ident $(,)?) => {
        nonzero_integer! {
            #[stable(feature = "nonzero", since = "1.28.0")]
            Self = $Ty,
            Primitive = unsigned $Int,
            UnsignedPrimitive = $Int,
            leading_zeros_test = concat!(stringify!($Int), "::MAX"),
        }
    };

    (Self = $Ty:ident, Primitive = signed $Int:ident, $($rest:tt)*) => {
        nonzero_integer! {
            #[stable(feature = "signed_nonzero", since = "1.34.0")]
            Self = $Ty,
            Primitive = signed $Int,
            $($rest)*
            leading_zeros_test = concat!("-1", stringify!($Int)),
        }
    };
}

macro_rules! nonzero_integer_signedness_dependent_impls {
    // Impls for unsigned nonzero types only.
    ($Ty:ident unsigned $Int:ty) => {
        #[stable(feature = "nonzero_div", since = "1.51.0")]
        impl Div<$Ty> for $Int {
            type Output = $Int;

            /// This operation rounds towards zero,
            /// truncating any fractional part of the exact result, and cannot panic.
            #[inline]
            fn div(self, other: $Ty) -> $Int {
                // SAFETY: div by zero is checked because `other` is a nonzero,
                // and MIN/-1 is checked because `self` is an unsigned int.
                unsafe { intrinsics::unchecked_div(self, other.get()) }
            }
        }

        #[stable(feature = "nonzero_div", since = "1.51.0")]
        impl Rem<$Ty> for $Int {
            type Output = $Int;

            /// This operation satisfies `n % d == n - (n / d) * d`, and cannot panic.
            #[inline]
            fn rem(self, other: $Ty) -> $Int {
                // SAFETY: rem by zero is checked because `other` is a nonzero,
                // and MIN/-1 is checked because `self` is an unsigned int.
                unsafe { intrinsics::unchecked_rem(self, other.get()) }
            }
        }
    };

    // Impls for signed nonzero types only.
    ($Ty:ident signed $Int:ty) => {
        #[stable(feature = "signed_nonzero_neg", since = "1.71.0")]
        impl Neg for $Ty {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                // SAFETY: negation of nonzero cannot yield zero values.
                unsafe { Self::new_unchecked(self.get().neg()) }
            }
        }

        forward_ref_unop! { impl Neg, neg for $Ty,
        #[stable(feature = "signed_nonzero_neg", since = "1.71.0")] }
    };
}

#[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5974
macro_rules! nonzero_integer_signedness_dependent_methods {
    // Associated items for unsigned nonzero types only.
    (
        Self = $Ty:ident,
        Primitive = unsigned $Int:ident,
        UnsignedPrimitive = $Uint:ty,
    ) => {
        /// The smallest value that can be represented by this non-zero
        /// integer type, 1.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::MIN.get(), 1", stringify!($Int), ");")]
        /// ```
        #[stable(feature = "nonzero_min_max", since = "1.70.0")]
        pub const MIN: Self = Self::new(1).unwrap();

        /// The largest value that can be represented by this non-zero
        /// integer type,
        #[doc = concat!("equal to [`", stringify!($Int), "::MAX`].")]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::MAX.get(), ", stringify!($Int), "::MAX);")]
        /// ```
        #[stable(feature = "nonzero_min_max", since = "1.70.0")]
        pub const MAX: Self = Self::new(<$Int>::MAX).unwrap();

        /// Adds an unsigned integer to a non-zero value.
        /// Checks for overflow and returns [`None`] on overflow.
        /// As a consequence, the result cannot wrap to zero.
        ///
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
        #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(Some(two), one.checked_add(1));
        /// assert_eq!(None, max.checked_add(1));
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_add(self, other: $Int) -> Option<Self> {
            if let Some(result) = self.get().checked_add(other) {
                // SAFETY:
                // - `checked_add` returns `None` on overflow
                // - `self` is non-zero
                // - the only way to get zero from an addition without overflow is for both
                //   sides to be zero
                //
                // So the result cannot be zero.
                Some(unsafe { Self::new_unchecked(result) })
            } else {
                None
            }
        }

        /// Adds an unsigned integer to a non-zero value.
        #[doc = concat!("Return [`", stringify!($Ty), "::MAX`] on overflow.")]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
        #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(two, one.saturating_add(1));
        /// assert_eq!(max, max.saturating_add(1));
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_add(self, other: $Int) -> Self {
            // SAFETY:
            // - `saturating_add` returns `u*::MAX` on overflow, which is non-zero
            // - `self` is non-zero
            // - the only way to get zero from an addition without overflow is for both
            //   sides to be zero
            //
            // So the result cannot be zero.
            unsafe { Self::new_unchecked(self.get().saturating_add(other)) }
        }

        /// Adds an unsigned integer to a non-zero value,
        /// assuming overflow cannot occur.
        /// Overflow is unchecked, and it is undefined behaviour to overflow
        /// *even if the result would wrap to a non-zero value*.
        /// The behaviour is undefined as soon as
        #[doc = concat!("`self + rhs > ", stringify!($Int), "::MAX`.")]
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(nonzero_ops)]
        ///
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
        ///
        /// assert_eq!(two, unsafe { one.unchecked_add(1) });
        /// # Some(())
        /// # }
        /// ```
        #[unstable(feature = "nonzero_ops", issue = "84186")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const unsafe fn unchecked_add(self, other: $Int) -> Self {
            // SAFETY: The caller ensures there is no overflow.
            unsafe { Self::new_unchecked(self.get().unchecked_add(other)) }
        }

        /// Returns the smallest power of two greater than or equal to n.
        /// Checks for overflow and returns [`None`]
        /// if the next power of two is greater than the type’s maximum value.
        /// As a consequence, the result cannot wrap to zero.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
        #[doc = concat!("let three = ", stringify!($Ty), "::new(3)?;")]
        #[doc = concat!("let four = ", stringify!($Ty), "::new(4)?;")]
        #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(Some(two), two.checked_next_power_of_two() );
        /// assert_eq!(Some(four), three.checked_next_power_of_two() );
        /// assert_eq!(None, max.checked_next_power_of_two() );
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_next_power_of_two(self) -> Option<Self> {
            if let Some(nz) = self.get().checked_next_power_of_two() {
                // SAFETY: The next power of two is positive
                // and overflow is checked.
                Some(unsafe { Self::new_unchecked(nz) })
            } else {
                None
            }
        }

        /// Returns the base 2 logarithm of the number, rounded down.
        ///
        /// This is the same operation as
        #[doc = concat!("[`", stringify!($Int), "::ilog2`],")]
        /// except that it has no failure cases to worry about
        /// since this value can never be zero.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(7).unwrap().ilog2(), 2);")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(8).unwrap().ilog2(), 3);")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(9).unwrap().ilog2(), 3);")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn ilog2(self) -> u32 {
            Self::BITS - 1 - self.leading_zeros()
        }

        /// Returns the base 10 logarithm of the number, rounded down.
        ///
        /// This is the same operation as
        #[doc = concat!("[`", stringify!($Int), "::ilog10`],")]
        /// except that it has no failure cases to worry about
        /// since this value can never be zero.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(99).unwrap().ilog10(), 1);")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(100).unwrap().ilog10(), 2);")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::new(101).unwrap().ilog10(), 2);")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn ilog10(self) -> u32 {
            super::int_log10::$Int(self.get())
        }

        /// Calculates the middle point of `self` and `rhs`.
        ///
        /// `midpoint(a, b)` is `(a + b) >> 1` as if it were performed in a
        /// sufficiently-large signed integral type. This implies that the result is
        /// always rounded towards negative infinity and that no overflow will ever occur.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(num_midpoint)]
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        ///
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let two = ", stringify!($Ty), "::new(2)?;")]
        #[doc = concat!("let four = ", stringify!($Ty), "::new(4)?;")]
        ///
        /// assert_eq!(one.midpoint(four), two);
        /// assert_eq!(four.midpoint(one), two);
        /// # Some(())
        /// # }
        /// ```
        #[unstable(feature = "num_midpoint", issue = "110840")]
        #[rustc_const_unstable(feature = "const_num_midpoint", issue = "110840")]
        #[rustc_allow_const_fn_unstable(const_num_midpoint)]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn midpoint(self, rhs: Self) -> Self {
            // SAFETY: The only way to get `0` with midpoint is to have two opposite or
            // near opposite numbers: (-5, 5), (0, 1), (0, 0) which is impossible because
            // of the unsignedness of this number and also because `Self` is guaranteed to
            // never being 0.
            unsafe { Self::new_unchecked(self.get().midpoint(rhs.get())) }
        }

        /// Returns `true` if and only if `self == (1 << k)` for some `k`.
        ///
        /// On many architectures, this function can perform better than `is_power_of_two()`
        /// on the underlying integer type, as special handling of zero can be avoided.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let eight = std::num::", stringify!($Ty), "::new(8).unwrap();")]
        /// assert!(eight.is_power_of_two());
        #[doc = concat!("let ten = std::num::", stringify!($Ty), "::new(10).unwrap();")]
        /// assert!(!ten.is_power_of_two());
        /// ```
        #[must_use]
        #[stable(feature = "nonzero_is_power_of_two", since = "1.59.0")]
        #[rustc_const_stable(feature = "nonzero_is_power_of_two", since = "1.59.0")]
        #[inline]
        pub const fn is_power_of_two(self) -> bool {
            // LLVM 11 normalizes `unchecked_sub(x, 1) & x == 0` to the implementation seen here.
            // On the basic x86-64 target, this saves 3 instructions for the zero check.
            // On x86_64 with BMI1, being nonzero lets it codegen to `BLSR`, which saves an instruction
            // compared to the `POPCNT` implementation on the underlying integer type.

            intrinsics::ctpop(self.get()) < 2
        }
    };

    // Associated items for signed nonzero types only.
    (
        Self = $Ty:ident,
        Primitive = signed $Int:ident,
        UnsignedNonZero = $Uty:ident,
        UnsignedPrimitive = $Uint:ty,
    ) => {
        /// The smallest value that can be represented by this non-zero
        /// integer type,
        #[doc = concat!("equal to [`", stringify!($Int), "::MIN`].")]
        ///
        /// Note: While most integer types are defined for every whole
        /// number between `MIN` and `MAX`, signed non-zero integers are
        /// a special case. They have a "gap" at 0.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::MIN.get(), ", stringify!($Int), "::MIN);")]
        /// ```
        #[stable(feature = "nonzero_min_max", since = "1.70.0")]
        pub const MIN: Self = Self::new(<$Int>::MIN).unwrap();

        /// The largest value that can be represented by this non-zero
        /// integer type,
        #[doc = concat!("equal to [`", stringify!($Int), "::MAX`].")]
        ///
        /// Note: While most integer types are defined for every whole
        /// number between `MIN` and `MAX`, signed non-zero integers are
        /// a special case. They have a "gap" at 0.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("assert_eq!(", stringify!($Ty), "::MAX.get(), ", stringify!($Int), "::MAX);")]
        /// ```
        #[stable(feature = "nonzero_min_max", since = "1.70.0")]
        pub const MAX: Self = Self::new(<$Int>::MAX).unwrap();

        /// Computes the absolute value of self.
        #[doc = concat!("See [`", stringify!($Int), "::abs`]")]
        /// for documentation on overflow behaviour.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
        ///
        /// assert_eq!(pos, pos.abs());
        /// assert_eq!(pos, neg.abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn abs(self) -> Self {
            // SAFETY: This cannot overflow to zero.
            unsafe { Self::new_unchecked(self.get().abs()) }
        }

        /// Checked absolute value.
        /// Checks for overflow and returns [`None`] if
        #[doc = concat!("`self == ", stringify!($Ty), "::MIN`.")]
        /// The result cannot be zero.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        ///
        /// assert_eq!(Some(pos), neg.checked_abs());
        /// assert_eq!(None, min.checked_abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_abs(self) -> Option<Self> {
            if let Some(nz) = self.get().checked_abs() {
                // SAFETY: absolute value of nonzero cannot yield zero values.
                Some(unsafe { Self::new_unchecked(nz) })
            } else {
                None
            }
        }

        /// Computes the absolute value of self,
        /// with overflow information, see
        #[doc = concat!("[`", stringify!($Int), "::overflowing_abs`].")]
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        ///
        /// assert_eq!((pos, false), pos.overflowing_abs());
        /// assert_eq!((pos, false), neg.overflowing_abs());
        /// assert_eq!((min, true), min.overflowing_abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_abs(self) -> (Self, bool) {
            let (nz, flag) = self.get().overflowing_abs();
            (
                // SAFETY: absolute value of nonzero cannot yield zero values.
                unsafe { Self::new_unchecked(nz) },
                flag,
            )
        }

        /// Saturating absolute value, see
        #[doc = concat!("[`", stringify!($Int), "::saturating_abs`].")]
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let min_plus = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN + 1)?;")]
        #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(pos, pos.saturating_abs());
        /// assert_eq!(pos, neg.saturating_abs());
        /// assert_eq!(max, min.saturating_abs());
        /// assert_eq!(max, min_plus.saturating_abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_abs(self) -> Self {
            // SAFETY: absolute value of nonzero cannot yield zero values.
            unsafe { Self::new_unchecked(self.get().saturating_abs()) }
        }

        /// Wrapping absolute value, see
        #[doc = concat!("[`", stringify!($Int), "::wrapping_abs`].")]
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        #[doc = concat!("# let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(pos, pos.wrapping_abs());
        /// assert_eq!(pos, neg.wrapping_abs());
        /// assert_eq!(min, min.wrapping_abs());
        /// assert_eq!(max, (-max).wrapping_abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_abs(self) -> Self {
            // SAFETY: absolute value of nonzero cannot yield zero values.
            unsafe { Self::new_unchecked(self.get().wrapping_abs()) }
        }

        /// Computes the absolute value of self
        /// without any wrapping or panicking.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        #[doc = concat!("# use std::num::", stringify!($Uty), ";")]
        ///
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let u_pos = ", stringify!($Uty), "::new(1)?;")]
        #[doc = concat!("let i_pos = ", stringify!($Ty), "::new(1)?;")]
        #[doc = concat!("let i_neg = ", stringify!($Ty), "::new(-1)?;")]
        #[doc = concat!("let i_min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let u_max = ", stringify!($Uty), "::new(",
                        stringify!($Uint), "::MAX / 2 + 1)?;")]
        ///
        /// assert_eq!(u_pos, i_pos.unsigned_abs());
        /// assert_eq!(u_pos, i_neg.unsigned_abs());
        /// assert_eq!(u_max, i_min.unsigned_abs());
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "nonzero_checked_ops", since = "1.64.0")]
        #[rustc_const_stable(feature = "const_nonzero_checked_ops", since = "1.64.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn unsigned_abs(self) -> $Uty {
            // SAFETY: absolute value of nonzero cannot yield zero values.
            unsafe { $Uty::new_unchecked(self.get().unsigned_abs()) }
        }

        /// Returns `true` if `self` is positive and `false` if the
        /// number is negative.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        ///
        /// assert!(pos_five.is_positive());
        /// assert!(!neg_five.is_positive());
        /// # Some(())
        /// # }
        /// ```
        #[must_use]
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn is_positive(self) -> bool {
            self.get().is_positive()
        }

        /// Returns `true` if `self` is negative and `false` if the
        /// number is positive.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        ///
        /// assert!(neg_five.is_negative());
        /// assert!(!pos_five.is_negative());
        /// # Some(())
        /// # }
        /// ```
        #[must_use]
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn is_negative(self) -> bool {
            self.get().is_negative()
        }

        /// Checked negation. Computes `-self`,
        #[doc = concat!("returning `None` if `self == ", stringify!($Ty), "::MIN`.")]
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        ///
        /// assert_eq!(pos_five.checked_neg(), Some(neg_five));
        /// assert_eq!(min.checked_neg(), None);
        /// # Some(())
        /// # }
        /// ```
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn checked_neg(self) -> Option<Self> {
            if let Some(result) = self.get().checked_neg() {
                // SAFETY: negation of nonzero cannot yield zero values.
                return Some(unsafe { Self::new_unchecked(result) });
            }
            None
        }

        /// Negates self, overflowing if this is equal to the minimum value.
        ///
        #[doc = concat!("See [`", stringify!($Int), "::overflowing_neg`]")]
        /// for documentation on overflow behaviour.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        ///
        /// assert_eq!(pos_five.overflowing_neg(), (neg_five, false));
        /// assert_eq!(min.overflowing_neg(), (min, true));
        /// # Some(())
        /// # }
        /// ```
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn overflowing_neg(self) -> (Self, bool) {
            let (result, overflow) = self.get().overflowing_neg();
            // SAFETY: negation of nonzero cannot yield zero values.
            ((unsafe { Self::new_unchecked(result) }), overflow)
        }

        /// Saturating negation. Computes `-self`,
        #[doc = concat!("returning [`", stringify!($Ty), "::MAX`]")]
        #[doc = concat!("if `self == ", stringify!($Ty), "::MIN`")]
        /// instead of overflowing.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let min_plus_one = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN + 1)?;")]
        #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MAX)?;")]
        ///
        /// assert_eq!(pos_five.saturating_neg(), neg_five);
        /// assert_eq!(min.saturating_neg(), max);
        /// assert_eq!(max.saturating_neg(), min_plus_one);
        /// # Some(())
        /// # }
        /// ```
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn saturating_neg(self) -> Self {
            if let Some(result) = self.checked_neg() {
                return result;
            }
            Self::MAX
        }

        /// Wrapping (modular) negation. Computes `-self`, wrapping around at the boundary
        /// of the type.
        ///
        #[doc = concat!("See [`", stringify!($Int), "::wrapping_neg`]")]
        /// for documentation on overflow behaviour.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = ", stringify!($Ty), "::new(5)?;")]
        #[doc = concat!("let neg_five = ", stringify!($Ty), "::new(-5)?;")]
        #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                        stringify!($Int), "::MIN)?;")]
        ///
        /// assert_eq!(pos_five.wrapping_neg(), neg_five);
        /// assert_eq!(min.wrapping_neg(), min);
        /// # Some(())
        /// # }
        /// ```
        #[inline]
        #[stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        #[rustc_const_stable(feature = "nonzero_negation_ops", since = "1.71.0")]
        pub const fn wrapping_neg(self) -> Self {
            let result = self.get().wrapping_neg();
            // SAFETY: negation of nonzero cannot yield zero values.
            unsafe { Self::new_unchecked(result) }
        }
    };
}

// Use this when the generated code should differ between signed and unsigned types.
macro_rules! sign_dependent_expr {
    (signed ? if signed { $signed_case:expr } if unsigned { $unsigned_case:expr } ) => {
        $signed_case
    };
    (unsigned ? if signed { $signed_case:expr } if unsigned { $unsigned_case:expr } ) => {
        $unsigned_case
    };
}

nonzero_integer! {
    Self = NonZeroU8,
    Primitive = unsigned u8,
}

nonzero_integer! {
    Self = NonZeroU16,
    Primitive = unsigned u16,
}

nonzero_integer! {
    Self = NonZeroU32,
    Primitive = unsigned u32,
}

nonzero_integer! {
    Self = NonZeroU64,
    Primitive = unsigned u64,
}

nonzero_integer! {
    Self = NonZeroU128,
    Primitive = unsigned u128,
}

nonzero_integer! {
    Self = NonZeroUsize,
    Primitive = unsigned usize,
}

nonzero_integer! {
    Self = NonZeroI8,
    Primitive = signed i8,
    UnsignedNonZero = NonZeroU8,
    UnsignedPrimitive = u8,
}

nonzero_integer! {
    Self = NonZeroI16,
    Primitive = signed i16,
    UnsignedNonZero = NonZeroU16,
    UnsignedPrimitive = u16,
}

nonzero_integer! {
    Self = NonZeroI32,
    Primitive = signed i32,
    UnsignedNonZero = NonZeroU32,
    UnsignedPrimitive = u32,
}

nonzero_integer! {
    Self = NonZeroI64,
    Primitive = signed i64,
    UnsignedNonZero = NonZeroU64,
    UnsignedPrimitive = u64,
}

nonzero_integer! {
    Self = NonZeroI128,
    Primitive = signed i128,
    UnsignedNonZero = NonZeroU128,
    UnsignedPrimitive = u128,
}

nonzero_integer! {
    Self = NonZeroIsize,
    Primitive = signed isize,
    UnsignedNonZero = NonZeroUsize,
    UnsignedPrimitive = usize,
}
