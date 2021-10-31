//! Definitions of integer that is known not to equal zero.

use crate::fmt;
use crate::ops::{BitOr, BitOrAssign, Div, Rem};
use crate::str::FromStr;

use super::from_str_radix;
use super::{IntErrorKind, ParseIntError};
use crate::intrinsics;

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

macro_rules! nonzero_integers {
    ( $( #[$stability: meta] #[$const_new_unchecked_stability: meta] $Ty: ident($Int: ty); )+ ) => {
        $(
            /// An integer that is known not to equal zero.
            ///
            /// This enables some memory layout optimization.
            #[doc = concat!("For example, `Option<", stringify!($Ty), ">` is the same size as `", stringify!($Int), "`:")]
            ///
            /// ```rust
            /// use std::mem::size_of;
            #[doc = concat!("assert_eq!(size_of::<Option<core::num::", stringify!($Ty), ">>(), size_of::<", stringify!($Int), ">());")]
            /// ```
            #[$stability]
            #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
            #[repr(transparent)]
            #[rustc_layout_scalar_valid_range_start(1)]
            #[rustc_nonnull_optimization_guaranteed]
            pub struct $Ty($Int);

            impl $Ty {
                /// Creates a non-zero without checking whether the value is non-zero.
                /// This results in undefined behaviour if the value is zero.
                ///
                /// # Safety
                ///
                /// The value must not be zero.
                #[$stability]
                #[$const_new_unchecked_stability]
                #[must_use]
                #[inline]
                pub const unsafe fn new_unchecked(n: $Int) -> Self {
                    // SAFETY: this is guaranteed to be safe by the caller.
                    unsafe { Self(n) }
                }

                /// Creates a non-zero if the given value is not zero.
                #[$stability]
                #[rustc_const_stable(feature = "const_nonzero_int_methods", since = "1.47.0")]
                #[must_use]
                #[inline]
                pub const fn new(n: $Int) -> Option<Self> {
                    if n != 0 {
                        // SAFETY: we just checked that there's no `0`
                        Some(unsafe { Self(n) })
                    } else {
                        None
                    }
                }

                /// Returns the value as a primitive type.
                #[$stability]
                #[inline]
                #[rustc_const_stable(feature = "nonzero", since = "1.34.0")]
                pub const fn get(self) -> $Int {
                    self.0
                }

            }

            #[stable(feature = "from_nonzero", since = "1.31.0")]
            #[rustc_const_unstable(feature = "const_num_from_num", issue = "87852")]
            impl const From<$Ty> for $Int {
                #[doc = concat!("Converts a `", stringify!($Ty), "` into an `", stringify!($Int), "`")]
                #[inline]
                fn from(nonzero: $Ty) -> Self {
                    nonzero.0
                }
            }

            #[stable(feature = "nonzero_bitor", since = "1.45.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const BitOr for $Ty {
                type Output = Self;
                #[inline]
                fn bitor(self, rhs: Self) -> Self::Output {
                    // SAFETY: since `self` and `rhs` are both nonzero, the
                    // result of the bitwise-or will be nonzero.
                    unsafe { $Ty::new_unchecked(self.get() | rhs.get()) }
                }
            }

            #[stable(feature = "nonzero_bitor", since = "1.45.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const BitOr<$Int> for $Ty {
                type Output = Self;
                #[inline]
                fn bitor(self, rhs: $Int) -> Self::Output {
                    // SAFETY: since `self` is nonzero, the result of the
                    // bitwise-or will be nonzero regardless of the value of
                    // `rhs`.
                    unsafe { $Ty::new_unchecked(self.get() | rhs) }
                }
            }

            #[stable(feature = "nonzero_bitor", since = "1.45.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const BitOr<$Ty> for $Int {
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
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const BitOrAssign for $Ty {
                #[inline]
                fn bitor_assign(&mut self, rhs: Self) {
                    *self = *self | rhs;
                }
            }

            #[stable(feature = "nonzero_bitor", since = "1.45.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const BitOrAssign<$Int> for $Ty {
                #[inline]
                fn bitor_assign(&mut self, rhs: $Int) {
                    *self = *self | rhs;
                }
            }

            impl_nonzero_fmt! {
                #[$stability] (Debug, Display, Binary, Octal, LowerHex, UpperHex) for $Ty
            }
        )+
    }
}

nonzero_integers! {
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroU8(u8);
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroU16(u16);
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroU32(u32);
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroU64(u64);
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroU128(u128);
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")] NonZeroUsize(usize);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI8(i8);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI16(i16);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI32(i32);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI64(i64);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI128(i128);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroIsize(isize);
}

macro_rules! from_str_radix_nzint_impl {
    ($($t:ty)*) => {$(
        #[stable(feature = "nonzero_parse", since = "1.35.0")]
        impl FromStr for $t {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, Self::Err> {
                Self::new(from_str_radix(src, 10)?)
                    .ok_or(ParseIntError {
                        kind: IntErrorKind::Zero
                    })
            }
        }
    )*}
}

from_str_radix_nzint_impl! { NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128 NonZeroUsize
NonZeroI8 NonZeroI16 NonZeroI32 NonZeroI64 NonZeroI128 NonZeroIsize }

macro_rules! nonzero_leading_trailing_zeros {
    ( $( $Ty: ident($Uint: ty) , $LeadingTestExpr:expr ;)+ ) => {
        $(
            impl $Ty {
                /// Returns the number of leading zeros in the binary representation of `self`.
                ///
                /// On many architectures, this function can perform better than `leading_zeros()` on the underlying integer type, as special handling of zero can be avoided.
                ///
                /// # Examples
                ///
                /// Basic usage:
                ///
                /// ```
                #[doc = concat!("let n = std::num::", stringify!($Ty), "::new(", stringify!($LeadingTestExpr), ").unwrap();")]
                ///
                /// assert_eq!(n.leading_zeros(), 0);
                /// ```
                #[stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
                #[rustc_const_stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn leading_zeros(self) -> u32 {
                    // SAFETY: since `self` can not be zero it is safe to call ctlz_nonzero
                    unsafe { intrinsics::ctlz_nonzero(self.0 as $Uint) as u32 }
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
                    // SAFETY: since `self` can not be zero it is safe to call cttz_nonzero
                    unsafe { intrinsics::cttz_nonzero(self.0 as $Uint) as u32 }
                }

            }
        )+
    }
}

nonzero_leading_trailing_zeros! {
    NonZeroU8(u8), u8::MAX;
    NonZeroU16(u16), u16::MAX;
    NonZeroU32(u32), u32::MAX;
    NonZeroU64(u64), u64::MAX;
    NonZeroU128(u128), u128::MAX;
    NonZeroUsize(usize), usize::MAX;
    NonZeroI8(u8), -1i8;
    NonZeroI16(u16), -1i16;
    NonZeroI32(u32), -1i32;
    NonZeroI64(u64), -1i64;
    NonZeroI128(u128), -1i128;
    NonZeroIsize(usize), -1isize;
}

macro_rules! nonzero_integers_div {
    ( $( $Ty: ident($Int: ty); )+ ) => {
        $(
            #[stable(feature = "nonzero_div", since = "1.51.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const Div<$Ty> for $Int {
                type Output = $Int;
                /// This operation rounds towards zero,
                /// truncating any fractional part of the exact result, and cannot panic.
                #[inline]
                fn div(self, other: $Ty) -> $Int {
                    // SAFETY: div by zero is checked because `other` is a nonzero,
                    // and MIN/-1 is checked because `self` is an unsigned int.
                    unsafe { crate::intrinsics::unchecked_div(self, other.get()) }
                }
            }

            #[stable(feature = "nonzero_div", since = "1.51.0")]
            #[rustc_const_unstable(feature = "const_ops", issue = "90080")]
            impl const Rem<$Ty> for $Int {
                type Output = $Int;
                /// This operation satisfies `n % d == n - (n / d) * d`, and cannot panic.
                #[inline]
                fn rem(self, other: $Ty) -> $Int {
                    // SAFETY: rem by zero is checked because `other` is a nonzero,
                    // and MIN/-1 is checked because `self` is an unsigned int.
                    unsafe { crate::intrinsics::unchecked_rem(self, other.get()) }
                }
            }
        )+
    }
}

nonzero_integers_div! {
    NonZeroU8(u8);
    NonZeroU16(u16);
    NonZeroU32(u32);
    NonZeroU64(u64);
    NonZeroU128(u128);
    NonZeroUsize(usize);
}

// A bunch of methods for unsigned nonzero types only.
macro_rules! nonzero_unsigned_operations {
    ( $( $Ty: ident($Int: ty); )+ ) => {
        $(
            impl $Ty {
                /// Add an unsigned integer to a non-zero value.
                /// Check for overflow and return [`None`] on overflow
                /// As a consequence, the result cannot wrap to zero.
                ///
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn checked_add(self, other: $Int) -> Option<$Ty> {
                    if let Some(result) = self.get().checked_add(other) {
                        // SAFETY: $Int::checked_add returns None on overflow
                        // so the result cannot be zero.
                        Some(unsafe { $Ty::new_unchecked(result) })
                    } else {
                        None
                    }
                }

                /// Add an unsigned integer to a non-zero value.
                #[doc = concat!("Return [`", stringify!($Int), "::MAX`] on overflow.")]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn saturating_add(self, other: $Int) -> $Ty {
                    // SAFETY: $Int::saturating_add returns $Int::MAX on overflow
                    // so the result cannot be zero.
                    unsafe { $Ty::new_unchecked(self.get().saturating_add(other)) }
                }

                /// Add an unsigned integer to a non-zero value,
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
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                pub const unsafe fn unchecked_add(self, other: $Int) -> $Ty {
                    // SAFETY: The caller ensures there is no overflow.
                    unsafe { $Ty::new_unchecked(self.get().unchecked_add(other)) }
                }

                /// Returns the smallest power of two greater than or equal to n.
                /// Check for overflow and return [`None`]
                /// if the next power of two is greater than the type’s maximum value.
                /// As a consequence, the result cannot wrap to zero.
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn checked_next_power_of_two(self) -> Option<$Ty> {
                    if let Some(nz) = self.get().checked_next_power_of_two() {
                        // SAFETY: The next power of two is positive
                        // and overflow is checked.
                        Some(unsafe { $Ty::new_unchecked(nz) })
                    } else {
                        None
                    }
                }
            }
        )+
    }
}

nonzero_unsigned_operations! {
    NonZeroU8(u8);
    NonZeroU16(u16);
    NonZeroU32(u32);
    NonZeroU64(u64);
    NonZeroU128(u128);
    NonZeroUsize(usize);
}

// A bunch of methods for signed nonzero types only.
macro_rules! nonzero_signed_operations {
    ( $( $Ty: ident($Int: ty) -> $Uty: ident($Uint: ty); )+ ) => {
        $(
            impl $Ty {
                /// Computes the absolute value of self.
                #[doc = concat!("See [`", stringify!($Int), "::abs`]")]
                /// for documentation on overflow behaviour.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn abs(self) -> $Ty {
                    // SAFETY: This cannot overflow to zero.
                    unsafe { $Ty::new_unchecked(self.get().abs()) }
                }

                /// Checked absolute value.
                /// Check for overflow and returns [`None`] if
                #[doc = concat!("`self == ", stringify!($Int), "::MIN`.")]
                /// The result cannot be zero.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn checked_abs(self) -> Option<$Ty> {
                    if let Some(nz) = self.get().checked_abs() {
                        // SAFETY: absolute value of nonzero cannot yield zero values.
                        Some(unsafe { $Ty::new_unchecked(nz) })
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
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn overflowing_abs(self) -> ($Ty, bool) {
                    let (nz, flag) = self.get().overflowing_abs();
                    (
                        // SAFETY: absolute value of nonzero cannot yield zero values.
                        unsafe { $Ty::new_unchecked(nz) },
                        flag,
                    )
                }

                /// Saturating absolute value, see
                #[doc = concat!("[`", stringify!($Int), "::saturating_abs`].")]
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn saturating_abs(self) -> $Ty {
                    // SAFETY: absolute value of nonzero cannot yield zero values.
                    unsafe { $Ty::new_unchecked(self.get().saturating_abs()) }
                }

                /// Wrapping absolute value, see
                #[doc = concat!("[`", stringify!($Int), "::wrapping_abs`].")]
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::new(-1)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::new(",
                                stringify!($Int), "::MIN)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::new(",
                                stringify!($Int), "::MAX)?;")]
                ///
                /// assert_eq!(pos, pos.wrapping_abs());
                /// assert_eq!(pos, neg.wrapping_abs());
                /// assert_eq!(min, min.wrapping_abs());
                /// # // FIXME: add once Neg is implemented?
                /// # // assert_eq!(max, (-max).wrapping_abs());
                /// # Some(())
                /// # }
                /// ```
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn wrapping_abs(self) -> $Ty {
                    // SAFETY: absolute value of nonzero cannot yield zero values.
                    unsafe { $Ty::new_unchecked(self.get().wrapping_abs()) }
                }

                /// Computes the absolute value of self
                /// without any wrapping or panicking.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_ops)]
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn unsigned_abs(self) -> $Uty {
                    // SAFETY: absolute value of nonzero cannot yield zero values.
                    unsafe { $Uty::new_unchecked(self.get().unsigned_abs()) }
                }
            }
        )+
    }
}

nonzero_signed_operations! {
    NonZeroI8(i8) -> NonZeroU8(u8);
    NonZeroI16(i16) -> NonZeroU16(u16);
    NonZeroI32(i32) -> NonZeroU32(u32);
    NonZeroI64(i64) -> NonZeroU64(u64);
    NonZeroI128(i128) -> NonZeroU128(u128);
    NonZeroIsize(isize) -> NonZeroUsize(usize);
}

// A bunch of methods for both signed and unsigned nonzero types.
macro_rules! nonzero_unsigned_signed_operations {
    ( $( $signedness:ident $Ty: ident($Int: ty); )+ ) => {
        $(
            impl $Ty {
                /// Multiply two non-zero integers together.
                /// Check for overflow and return [`None`] on overflow.
                /// As a consequence, the result cannot wrap to zero.
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn checked_mul(self, other: $Ty) -> Option<$Ty> {
                    if let Some(result) = self.get().checked_mul(other.get()) {
                        // SAFETY: checked_mul returns None on overflow
                        // and `other` is also non-null
                        // so the result cannot be zero.
                        Some(unsafe { $Ty::new_unchecked(result) })
                    } else {
                        None
                    }
                }

                /// Multiply two non-zero integers together.
                #[doc = concat!("Return [`", stringify!($Int), "::MAX`] on overflow.")]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn saturating_mul(self, other: $Ty) -> $Ty {
                    // SAFETY: saturating_mul returns u*::MAX on overflow
                    // and `other` is also non-null
                    // so the result cannot be zero.
                    unsafe { $Ty::new_unchecked(self.get().saturating_mul(other.get())) }
                }

                /// Multiply two non-zero integers together,
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
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                pub const unsafe fn unchecked_mul(self, other: $Ty) -> $Ty {
                    // SAFETY: The caller ensures there is no overflow.
                    unsafe { $Ty::new_unchecked(self.get().unchecked_mul(other.get())) }
                }

                /// Raise non-zero value to an integer power.
                /// Check for overflow and return [`None`] on overflow.
                /// As a consequence, the result cannot wrap to zero.
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn checked_pow(self, other: u32) -> Option<$Ty> {
                    if let Some(result) = self.get().checked_pow(other) {
                        // SAFETY: checked_pow returns None on overflow
                        // so the result cannot be zero.
                        Some(unsafe { $Ty::new_unchecked(result) })
                    } else {
                        None
                    }
                }

                /// Raise non-zero value to an integer power.
                #[doc = sign_dependent_expr!{
                    $signedness ?
                    if signed {
                        concat!("Return [`", stringify!($Int), "::MIN`] ",
                                    "or [`", stringify!($Int), "::MAX`] on overflow.")
                    }
                    if unsigned {
                        concat!("Return [`", stringify!($Int), "::MAX`] on overflow.")
                    }
                }]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(nonzero_ops)]
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                ///
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
                #[unstable(feature = "nonzero_ops", issue = "84186")]
                #[must_use = "this returns the result of the operation, \
                              without modifying the original"]
                #[inline]
                pub const fn saturating_pow(self, other: u32) -> $Ty {
                    // SAFETY: saturating_pow returns u*::MAX on overflow
                    // so the result cannot be zero.
                    unsafe { $Ty::new_unchecked(self.get().saturating_pow(other)) }
                }
            }
        )+
    }
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

nonzero_unsigned_signed_operations! {
    unsigned NonZeroU8(u8);
    unsigned NonZeroU16(u16);
    unsigned NonZeroU32(u32);
    unsigned NonZeroU64(u64);
    unsigned NonZeroU128(u128);
    unsigned NonZeroUsize(usize);
    signed NonZeroI8(i8);
    signed NonZeroI16(i16);
    signed NonZeroI32(i32);
    signed NonZeroI64(i64);
    signed NonZeroI128(i128);
    signed NonZeroIsize(isize);
}

macro_rules! nonzero_unsigned_is_power_of_two {
    ( $( $Ty: ident )+ ) => {
        $(
            impl $Ty {

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
                /// #![feature(nonzero_is_power_of_two)]
                ///
                #[doc = concat!("let eight = std::num::", stringify!($Ty), "::new(8).unwrap();")]
                /// assert!(eight.is_power_of_two());
                #[doc = concat!("let ten = std::num::", stringify!($Ty), "::new(10).unwrap();")]
                /// assert!(!ten.is_power_of_two());
                /// ```
                #[must_use]
                #[unstable(feature = "nonzero_is_power_of_two", issue = "81106")]
                #[inline]
                pub const fn is_power_of_two(self) -> bool {
                    // LLVM 11 normalizes `unchecked_sub(x, 1) & x == 0` to the implementation seen here.
                    // On the basic x86-64 target, this saves 3 instructions for the zero check.
                    // On x86_64 with BMI1, being nonzero lets it codegen to `BLSR`, which saves an instruction
                    // compared to the `POPCNT` implementation on the underlying integer type.

                    intrinsics::ctpop(self.get()) < 2
                }

            }
        )+
    }
}

nonzero_unsigned_is_power_of_two! { NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128 NonZeroUsize }
