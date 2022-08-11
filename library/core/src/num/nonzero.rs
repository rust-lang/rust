//! Definitions of integer that is known not to equal zero.

use crate::cmp::{Ord, Ordering, PartialEq, PartialOrd};
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::marker::{Destruct, StructuralPartialEq};
use crate::ops::{BitOr, BitOrAssign, Div, Neg, Rem};
use crate::str::FromStr;

use super::from_str_radix;
use super::{IntErrorKind, ParseIntError};
use crate::intrinsics;

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
pub trait ZeroablePrimitive: Sized + Copy + private::Sealed {
    /// Type from which a `NonZero<T>` is constructed, usually also `T`.
    type Input = Self;

    #[doc(hidden)]
    fn input_is_zero(n: &Self::Input) -> bool;
    #[doc(hidden)]
    fn from_input(n: Self::Input) -> Self;
}

macro_rules! impl_zeroable_primitive_int {
    ($($Ty:ident,)*) => {
        $(
            #[unstable(
                feature = "nonzero_internals",
                reason = "implementation detail which may disappear or be replaced at any time",
                issue = "none"
            )]
            impl const private::Sealed for $Ty {}

            #[unstable(
                feature = "nonzero_internals",
                reason = "implementation detail which may disappear or be replaced at any time",
                issue = "none"
            )]
            impl const ZeroablePrimitive for $Ty {
                #[inline(always)]
                fn input_is_zero(n: &Self::Input) -> bool {
                    *n == 0
                }

                #[inline(always)]
                fn from_input(n: Self::Input) -> Self {
                    n
                }
            }
        )+
    }
}

impl_zeroable_primitive_int!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,);

#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
impl<T: ?Sized> const private::Sealed for *const T {}

#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
impl<T: ?Sized> const ZeroablePrimitive for *const T {
    type Input = *mut T;

    #[inline(always)]
    fn input_is_zero(n: &Self::Input) -> bool {
        n.is_null()
    }

    #[inline(always)]
    fn from_input(n: Self::Input) -> Self {
        n as *const T
    }
}

/// A value that is known not to equal zero.
///
/// This enables some memory layout optimization.
/// For example, `Option<NonZero<u32>>` is the same size as `u32`:
///
/// ```rust
/// #![feature(generic_nonzero)]
///
/// use core::mem::size_of;
/// assert_eq!(size_of::<Option<core::num::NonZero<u32>>>(), size_of::<u32>());
/// ```
#[unstable(feature = "generic_nonzero", issue = "82363")]
#[derive(Copy, Eq)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
#[rustc_diagnostic_item = "NonZero"]
pub struct NonZero<T: ZeroablePrimitive>(T);

#[unstable(feature = "generic_nonzero", issue = "82363")]
impl<T: ZeroablePrimitive> Clone for NonZero<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { Self(self.0) }
    }
}

#[unstable(feature = "generic_nonzero", issue = "82363")]
impl<T: ZeroablePrimitive> PartialEq for NonZero<T>
where
    T: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 == other.0 }
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 != other.0 }
    }
}

#[stable(feature = "nonzero", since = "1.28.0")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> StructuralPartialEq for NonZero<T> {}

#[unstable(feature = "generic_nonzero", issue = "82363")]
impl<T: ZeroablePrimitive> PartialOrd for NonZero<T>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0.partial_cmp(&other.0) }
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 < other.0 }
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 <= other.0 }
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 > other.0 }
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0 >= other.0 }
    }
}

#[unstable(feature = "generic_nonzero", issue = "82363")]
impl<T: ZeroablePrimitive> Ord for NonZero<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0.cmp(&other.0) }
    }

    fn max(self, other: Self) -> Self
    where
        T: ~const Destruct,
    {
        // SAFETY: The maximum of two non-zero values is still non-zero.
        unsafe { Self(self.0.max(other.0)) }
    }

    fn min(self, other: Self) -> Self
    where
        T: ~const Destruct,
    {
        // SAFETY: The minimum of two non-zero values is still non-zero.
        unsafe { Self(self.0.min(other.0)) }
    }

    fn clamp(self, min: Self, max: Self) -> Self
    where
        T: ~const Destruct + PartialOrd,
    {
        // SAFETY: A non-zero value clamped between two non-zero values is still non-zero.
        unsafe { Self(self.0.clamp(min.0, max.0)) }
    }
}

#[unstable(feature = "generic_nonzero", issue = "82363")]
impl<T: ZeroablePrimitive> Hash for NonZero<T>
where
    T: Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        // SAFETY: A `NonZero` is guaranteed to only contain non-zero values.
        unsafe { self.0.hash(state) }
    }
}

impl<T: ZeroablePrimitive> NonZero<T> {
    /// Creates a non-zero without checking whether the value is non-zero.
    /// This results in undefined behaviour if the value is zero.
    ///
    /// # Safety
    ///
    /// The value must not be zero.
    #[stable(feature = "generic_nonzero_new", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "generic_nonzero_new", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_allow_const_fn_unstable(const_refs_to_cell)]
    #[must_use]
    #[inline]
    pub const unsafe fn new_unchecked(v: T::Input) -> Self
    where
        T: ~const ZeroablePrimitive,
        T::Input: ~const Destruct,
    {
        // SAFETY: This is guaranteed to be safe by the caller.
        unsafe {
            // FIXME: Make `assert_unsafe_precondition` work with `~const` bound.
            #[cfg(debug_assertions)]
            {
                let is_zero = T::input_is_zero(&v);
                core::intrinsics::assert_unsafe_precondition!(
                    "NonZero::new_unchecked requires a non-zero argument",
                    (is_zero: bool) => !is_zero
                );
            }

            Self(T::from_input(v))
        }
    }

    /// Creates a non-zero if the given value is not zero.
    #[stable(feature = "generic_nonzero_new", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "generic_nonzero_new", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_allow_const_fn_unstable(const_refs_to_cell)]
    #[must_use]
    #[inline]
    pub const fn new(v: T::Input) -> Option<Self>
    where
        T: ~const ZeroablePrimitive,
        T::Input: ~const Destruct,
    {
        if !T::input_is_zero(&v) {
            // SAFETY: We just checked that `n` is not 0.
            Some(unsafe { Self(T::from_input(v)) })
        } else {
            None
        }
    }

    /// Returns the value as a primitive type.
    #[stable(feature = "generic_nonzero_get", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "generic_nonzero_get", since = "CURRENT_RUSTC_VERSION")]
    #[must_use]
    #[inline]
    pub const fn get(self) -> T {
        self.0
    }
}

#[stable(feature = "from_generic_nonzero", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_num_from_num", issue = "87852")]
impl<T: ZeroablePrimitive> From<NonZero<T>> for T {
    /// Converts a `NonZero::<T>` into a `T`.
    #[inline]
    fn from(nonzero: NonZero<Self>) -> Self {
        nonzero.get()
    }
}

#[stable(feature = "generic_nonzero_bitor", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> BitOr<T> for NonZero<T>
where
    T: BitOr<Output = T>,
{
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: T) -> Self::Output {
        // SAFETY: Since `self` is non-zero, the result of the bitwise-or
        // will be non-zero regardless of the value of `rhs`.
        unsafe { Self(self.0 | rhs) }
    }
}

#[stable(feature = "generic_nonzero_bitor", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> BitOr for NonZero<T>
where
    T: BitOr<Output = T>,
{
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self | rhs.0
    }
}

#[stable(feature = "generic_nonzero_bitor", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> BitOr<NonZero<T>> for T
where
    T: BitOr<Output = T>,
{
    type Output = NonZero<T>;

    #[inline]
    fn bitor(self, rhs: NonZero<T>) -> Self::Output {
        rhs | self
    }
}

#[stable(feature = "generic_nonzero_bitor", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> BitOrAssign for NonZero<T>
where
    T: BitOr<Output = T>,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[stable(feature = "generic_nonzero_bitor", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_ops", issue = "90080")]
impl<T: ZeroablePrimitive> BitOrAssign<T> for NonZero<T>
where
    T: BitOr<Output = T>,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: T) {
        *self = *self | rhs;
    }
}

macro_rules! nonzero_fmt {
    (#[$stability:meta] ($($Trait:ident),+)) => {
        $(
            #[$stability]
            impl<T: ZeroablePrimitive> fmt::$Trait for NonZero<T>
            where
                T: fmt::$Trait,
            {
                #[inline]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::$Trait::fmt(&self.get(), f)
                }
            }
        )+
    };
}

nonzero_fmt!(
    #[stable(feature = "generic_nonzero_fmt", since = "CURRENT_RUSTC_VERSION")]
    (Debug, Display, Binary, Octal, LowerHex, UpperHex)
);

macro_rules! nonzero_alias {
    ($(
        #[$stability: meta] #[$const_new_unchecked_stability: meta]
        $(
          $vis:vis type $Alias:ident = $Ty:ident<$Int:ident>;
        )+
    )+) => {
        $($(
            #[doc = concat!("An `", stringify!($Int), "` that is known not to equal zero.")]
            ///
            /// This enables some memory layout optimization.
            #[doc = concat!("For example, `Option<", stringify!($Ty), "<", stringify!($Int), ">>` is the same size as `", stringify!($Int), "`:")]
            ///
            /// ```rust
            /// use core::mem::size_of;
            #[doc = concat!("assert_eq!(size_of::<Option<core::num::", stringify!($Alias), ">>(), size_of::<", stringify!($Int), ">());")]
            /// ```
            ///
            /// # Layout
            ///
            #[doc = concat!("`", stringify!($Ty), "<", stringify!($Int), ">` is guaranteed to have the same layout and bit validity as `", stringify!($Int), "`")]
            /// with the exception that `0` is not a valid instance.
            #[doc = concat!("`Option<", stringify!($Ty), "<", stringify!($Int), ">>` is guaranteed to be compatible with `", stringify!($Int), "`,")]
            /// including in FFI.
            #[$stability]
            #[rustc_diagnostic_item = stringify!($Alias)]
            $vis type $Alias = $Ty<$Int>;
        )+)+
    }
}

nonzero_alias! {
    #[stable(feature = "nonzero", since = "1.28.0")] #[rustc_const_stable(feature = "nonzero", since = "1.28.0")]
    pub type NonZeroU8 = NonZero<u8>;
    pub type NonZeroU16 = NonZero<u16>;
    pub type NonZeroU32 = NonZero<u32>;
    pub type NonZeroU64 = NonZero<u64>;
    pub type NonZeroU128 = NonZero<u128>;
    pub type NonZeroUsize = NonZero<usize>;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] #[rustc_const_stable(feature = "signed_nonzero", since = "1.34.0")]
    pub type NonZeroI8 = NonZero<i8>;
    pub type NonZeroI16 = NonZero<i16>;
    pub type NonZeroI32 = NonZero<i32>;
    pub type NonZeroI64 = NonZero<i64>;
    pub type NonZeroI128 = NonZero<i128>;
    pub type NonZeroIsize = NonZero<isize>;
}

macro_rules! from_str_radix_nzint_impl {
    ($($Ty:ident<$Int:ident>;)*) => {
        $(
            #[stable(feature = "nonzero_parse", since = "1.35.0")]
            impl FromStr for $Ty<$Int> {
                type Err = ParseIntError;

                fn from_str(src: &str) -> Result<Self, Self::Err> {
                    Self::new(from_str_radix(src, 10)?)
                        .ok_or(ParseIntError {
                            kind: IntErrorKind::Zero
                        })
                }
            }
        )*
    }
}

from_str_radix_nzint_impl! {
    NonZero<u8>;
    NonZero<i8>;
    NonZero<u16>;
    NonZero<i16>;
    NonZero<u32>;
    NonZero<i32>;
    NonZero<u64>;
    NonZero<i64>;
    NonZero<u128>;
    NonZero<i128>;
    NonZero<usize>;
    NonZero<isize>;
}

macro_rules! nonzero_leading_trailing_zeros {
    ($($Ty:ident<$Int:ident>, $Uint:ident, $LeadingTestExpr:expr;)+) => {
        $(
            impl $Ty<$Int> {
                /// Returns the number of leading zeros in the binary representation of `self`.
                ///
                /// On many architectures, this function can perform better than `leading_zeros()` on the underlying integer type, as special handling of zero can be avoided.
                ///
                /// # Examples
                ///
                /// Basic usage:
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("let n = core::num::", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($LeadingTestExpr), ").unwrap();")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("let n = core::num::", stringify!($Ty), "::<", stringify!($Int), ">::new(0b0101000).unwrap();")]
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
                    unsafe { intrinsics::cttz_nonzero(self.0 as $Uint) as u32 }
                }
            }
        )+
    }
}

nonzero_leading_trailing_zeros! {
    NonZero<u8>, u8, u8::MAX;
    NonZero<u16>, u16, u16::MAX;
    NonZero<u32>, u32, u32::MAX;
    NonZero<u64>, u64, u64::MAX;
    NonZero<u128>, u128, u128::MAX;
    NonZero<usize>, usize, usize::MAX;
    NonZero<i8>, u8, -1i8;
    NonZero<i16>, u16, -1i16;
    NonZero<i32>, u32, -1i32;
    NonZero<i64>, u64, -1i64;
    NonZero<i128>, u128, -1i128;
    NonZero<isize>, usize, -1isize;
}

macro_rules! nonzero_integers_div {
    ($($Ty:ident<$Int:ty>;)+) => {
        $(
            #[stable(feature = "nonzero_div", since = "1.51.0")]
            impl Div<$Ty<$Int>> for $Int {
                type Output = $Int;
                /// This operation rounds towards zero,
                /// truncating any fractional part of the exact result, and cannot panic.
                #[inline]
                fn div(self, other: $Ty<$Int>) -> $Int {
                    // SAFETY: div by zero is checked because `other` is a nonzero,
                    // and MIN/-1 is checked because `self` is an unsigned int.
                    unsafe { crate::intrinsics::unchecked_div(self, other.get()) }
                }
            }

            #[stable(feature = "nonzero_div", since = "1.51.0")]
            impl Rem<$Ty<$Int>> for $Int {
                type Output = $Int;
                /// This operation satisfies `n % d == n - (n / d) * d`, and cannot panic.
                #[inline]
                fn rem(self, other: $Ty<$Int>) -> $Int {
                    // SAFETY: rem by zero is checked because `other` is a nonzero,
                    // and MIN/-1 is checked because `self` is an unsigned int.
                    unsafe { crate::intrinsics::unchecked_rem(self, other.get()) }
                }
            }
        )+
    }
}

nonzero_integers_div! {
    NonZero<u8>;
    NonZero<u16>;
    NonZero<u32>;
    NonZero<u64>;
    NonZero<u128>;
    NonZero<usize>;
}

// A bunch of methods for unsigned nonzero types only.
macro_rules! nonzero_unsigned_operations {
    ($($Ty:ident<$Int:ident>;)+) => {
        $(
            impl $Ty<$Int> {
                /// Adds an unsigned integer to a non-zero value.
                /// Checks for overflow and returns [`None`] on overflow.
                /// As a consequence, the result cannot wrap to zero.
                ///
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let one = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                        // SAFETY: $Int::checked_add returns None on overflow
                        // so the result cannot be zero.
                        Some(unsafe { Self::new_unchecked(result) })
                    } else {
                        None
                    }
                }

                /// Adds an unsigned integer to a non-zero value.
                #[doc = concat!("Return [`", stringify!($Int), "::MAX`] on overflow.")]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let one = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                    // SAFETY: $Int::saturating_add returns $Int::MAX on overflow
                    // so the result cannot be zero.
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let one = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let three = ", stringify!($Ty), "::<", stringify!($Int), ">::new(3)?;")]
                #[doc = concat!("let four = ", stringify!($Ty), "::<", stringify!($Int), ">::new(4)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                        // SAFETY: The next power of two is positive and overflow is checked.
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(7).unwrap().ilog2(), 2);")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(8).unwrap().ilog2(), 3);")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(9).unwrap().ilog2(), 3);")]
                /// ```
                #[stable(feature = "int_log", since = "1.67.0")]
                #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
                #[must_use = "this returns the result of the operation, without modifying the original"]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(99).unwrap().ilog10(), 1);")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(100).unwrap().ilog10(), 2);")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::new(101).unwrap().ilog10(), 2);")]
                /// ```
                #[stable(feature = "int_log", since = "1.67.0")]
                #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
                #[must_use = "this returns the result of the operation, without modifying the original"]
                #[inline]
                pub const fn ilog10(self) -> u32 {
                    super::int_log10::$Int(self.0)
                }
            }
        )+
    }
}

nonzero_unsigned_operations! {
    NonZero<u8>;
    NonZero<u16>;
    NonZero<u32>;
    NonZero<u64>;
    NonZero<u128>;
    NonZero<usize>;
}

// A bunch of methods for signed nonzero types only.
macro_rules! nonzero_signed_operations {
    ($($Ty:ident<$Int:ty> -> $Uty:ident<$Uint:ty>;)+) => {
        $(
            impl $Ty<$Int> {
                /// Computes the absolute value of self.
                #[doc = concat!("See [`", stringify!($Int), "::abs`]")]
                /// for documentation on overflow behaviour.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
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
                #[doc = concat!("`self == ", stringify!($Int), "::MIN`.")]
                /// The result cannot be zero.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                #[doc = concat!("let min_plus = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN + 1)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let u_pos = ", stringify!($Uty), "::<", stringify!($Uint), ">::new(1)?;")]
                #[doc = concat!("let i_pos = ", stringify!($Ty), "::<", stringify!($Int), ">::new(1)?;")]
                #[doc = concat!("let i_neg = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-1)?;")]
                #[doc = concat!("let i_min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                #[doc = concat!("let u_max = ", stringify!($Uty), "::<", stringify!($Uint), ">::new(", stringify!($Uint), "::MAX / 2 + 1)?;")]
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
                pub const fn unsigned_abs(self) -> $Uty::<$Uint> {
                    // SAFETY: absolute value of nonzero cannot yield zero values.
                    unsafe { $Uty::<$Uint>::new_unchecked(self.get().unsigned_abs()) }
                }

                /// Returns `true` if `self` is positive and `false` if the
                /// number is negative.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use std::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                ///
                /// assert!(pos_five.is_positive());
                /// assert!(!neg_five.is_positive());
                /// # Some(())
                /// # }
                /// ```
                #[must_use]
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
                pub const fn is_positive(self) -> bool {
                    self.get().is_positive()
                }

                /// Returns `true` if `self` is negative and `false` if the
                /// number is positive.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                ///
                /// assert!(neg_five.is_negative());
                /// assert!(!pos_five.is_negative());
                /// # Some(())
                /// # }
                /// ```
                #[must_use]
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
                pub const fn is_negative(self) -> bool {
                    self.get().is_negative()
                }

                /// Checked negation. Computes `-self`, returning `None` if `self == i32::MIN`.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                ///
                /// assert_eq!(pos_five.checked_neg(), Some(neg_five));
                /// assert_eq!(min.checked_neg(), None);
                /// # Some(())
                /// # }
                /// ```
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
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
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                ///
                /// assert_eq!(pos_five.overflowing_neg(), (neg_five, false));
                /// assert_eq!(min.overflowing_neg(), (min, true));
                /// # Some(())
                /// # }
                /// ```
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
                pub const fn overflowing_neg(self) -> (Self, bool) {
                    let (result, overflow) = self.get().overflowing_neg();
                    // SAFETY: negation of nonzero cannot yield zero values.
                    (unsafe { Self::new_unchecked(result) }, overflow)
                }

                /// Saturating negation. Computes `-self`, returning `MAX` if
                /// `self == i32::MIN` instead of overflowing.
                ///
                /// # Example
                ///
                /// ```
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                #[doc = concat!("let min_plus_one = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN + 1)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
                ///
                /// assert_eq!(pos_five.saturating_neg(), neg_five);
                /// assert_eq!(min.saturating_neg(), max);
                /// assert_eq!(max.saturating_neg(), min_plus_one);
                /// # Some(())
                /// # }
                /// ```
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
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
                /// #![feature(nonzero_negation_ops)]
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let pos_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(5)?;")]
                #[doc = concat!("let neg_five = ", stringify!($Ty), "::<", stringify!($Int), ">::new(-5)?;")]
                #[doc = concat!("let min = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MIN)?;")]
                ///
                /// assert_eq!(pos_five.wrapping_neg(), neg_five);
                /// assert_eq!(min.wrapping_neg(), min);
                /// # Some(())
                /// # }
                /// ```
                #[inline]
                #[unstable(feature = "nonzero_negation_ops", issue = "102443")]
                pub const fn wrapping_neg(self) -> Self {
                    let result = self.get().wrapping_neg();
                    // SAFETY: negation of nonzero cannot yield zero values.
                    unsafe { Self::new_unchecked(result) }
                }
            }

            #[stable(feature = "signed_nonzero_neg", since = "CURRENT_RUSTC_VERSION")]
            impl Neg for $Ty<$Int> {
                type Output = Self;

                #[inline]
                fn neg(self) -> Self {
                    // SAFETY: negation of nonzero cannot yield zero values.
                    unsafe { Self::new_unchecked(self.get().neg()) }
                }
            }

            forward_ref_unop! { impl Neg, neg for $Ty<$Int>,
                #[stable(feature = "signed_nonzero_neg", since = "CURRENT_RUSTC_VERSION")] }
        )+
    }
}

nonzero_signed_operations! {
    NonZero<i8> -> NonZero<u8>;
    NonZero<i16> -> NonZero<u16>;
    NonZero<i32> -> NonZero<u32>;
    NonZero<i64> -> NonZero<u64>;
    NonZero<i128> -> NonZero<u128>;
    NonZero<isize> -> NonZero<usize>;
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

// A bunch of methods for both signed and unsigned nonzero types.
macro_rules! nonzero_unsigned_signed_operations {
    ( $( $signedness:ident $Ty:ident<$Int:ty>; )+ ) => {
        $(
            impl $Ty<$Int> {
                /// Multiplies two non-zero integers together.
                /// Checks for overflow and returns [`None`] on overflow.
                /// As a consequence, the result cannot wrap to zero.
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let four = ", stringify!($Ty), "::<", stringify!($Int), ">::new(4)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                        // SAFETY: checked_mul returns None on overflow
                        // and `other` is also non-null
                        // so the result cannot be zero.
                        Some(unsafe { Self::new_unchecked(result) })
                    } else {
                        None
                    }
                }

                /// Multiplies two non-zero integers together.
                #[doc = concat!("Return [`", stringify!($Int), "::MAX`] on overflow.")]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let four = ", stringify!($Ty), "::<", stringify!($Int), ">::new(4)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                    // SAFETY: `saturating_mul` returns `u*::MAX` on overflow
                    // and `other` is also non-null, so the result cannot be zero.
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let two = ", stringify!($Ty), "::<", stringify!($Int), ">::new(2)?;")]
                #[doc = concat!("let four = ", stringify!($Ty), "::<", stringify!($Int), ">::new(4)?;")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let three = ", stringify!($Ty), "::<", stringify!($Int), ">::new(3)?;")]
                #[doc = concat!("let twenty_seven = ", stringify!($Ty), "::<", stringify!($Int), ">::new(27)?;")]
                #[doc = concat!("let half_max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX / 2)?;")]
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
                        // SAFETY: checked_pow returns None on overflow
                        // so the result cannot be zero.
                        Some(unsafe { Self::new_unchecked(result) })
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                /// # fn main() { test().unwrap(); }
                /// # fn test() -> Option<()> {
                #[doc = concat!("let three = ", stringify!($Ty), "::<", stringify!($Int), ">::new(3)?;")]
                #[doc = concat!("let twenty_seven = ", stringify!($Ty), "::<", stringify!($Int), ">::new(27)?;")]
                #[doc = concat!("let max = ", stringify!($Ty), "::<", stringify!($Int), ">::new(", stringify!($Int), "::MAX)?;")]
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
                    // SAFETY: saturating_pow returns u*::MAX on overflow
                    // so the result cannot be zero.
                    unsafe { Self::new_unchecked(self.get().saturating_pow(other)) }
                }
            }
        )+
    }
}

nonzero_unsigned_signed_operations! {
    unsigned NonZero<u8>;
    unsigned NonZero<u16>;
    unsigned NonZero<u32>;
    unsigned NonZero<u64>;
    unsigned NonZero<u128>;
    unsigned NonZero<usize>;
    signed NonZero<i8>;
    signed NonZero<i16>;
    signed NonZero<i32>;
    signed NonZero<i64>;
    signed NonZero<i128>;
    signed NonZero<isize>;
}

macro_rules! nonzero_unsigned_is_power_of_two {
    ($($Ty:ident<$Int:ident>;)+) => {
        $(
            impl $Ty<$Int> {
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("let eight = core::num::", stringify!($Ty), "::<", stringify!($Int), ">::new(8).unwrap();")]
                /// assert!(eight.is_power_of_two());
                #[doc = concat!("let ten = core::num::", stringify!($Ty), "::<", stringify!($Int), ">::new(10).unwrap();")]
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
            }
        )+
    }
}

nonzero_unsigned_is_power_of_two! {
    NonZero<u8>;
    NonZero<u16>;
    NonZero<u32>;
    NonZero<u64>;
    NonZero<u128>;
    NonZero<usize>;
}

macro_rules! nonzero_min_max_unsigned {
    ($($Ty:ident<$Int:ident>;)+) => {
        $(
            impl $Ty<$Int> {
                /// The smallest value that can be represented by this non-zero
                /// integer type, 1.
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::MIN.get(), 1", stringify!($Int), ");")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::MAX.get(), ", stringify!($Int), "::MAX);")]
                /// ```
                #[stable(feature = "nonzero_min_max", since = "1.70.0")]
                pub const MAX: Self = Self::new(<$Int>::MAX).unwrap();
            }
        )+
    }
}

macro_rules! nonzero_min_max_signed {
    ($($Ty:ident<$Int:ident>;)+) => {
        $(
            impl $Ty<$Int> {
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::MIN.get(), ", stringify!($Int), "::MIN);")]
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
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::MAX.get(), ", stringify!($Int), "::MAX);")]
                /// ```
                #[stable(feature = "nonzero_min_max", since = "1.70.0")]
                pub const MAX: Self = Self::new(<$Int>::MAX).unwrap();
            }
        )+
    }
}

nonzero_min_max_unsigned! {
    NonZero<u8>;
    NonZero<u16>;
    NonZero<u32>;
    NonZero<u64>;
    NonZero<u128>;
    NonZero<usize>;
}

nonzero_min_max_signed! {
    NonZero<i8>;
    NonZero<i16>;
    NonZero<i32>;
    NonZero<i64>;
    NonZero<i128>;
    NonZero<isize>;
}

macro_rules! nonzero_bits {
    ($($Ty:ident<$Int:ty>;)+) => {
        $(
            impl $Ty<$Int> {
                /// The size of this non-zero integer type in bits.
                ///
                #[doc = concat!("This value is equal to [`", stringify!($Int), "::BITS`].")]
                ///
                /// # Examples
                ///
                /// ```
                /// #![feature(generic_nonzero)]
                ///
                #[doc = concat!("# use core::num::", stringify!($Ty), ";")]
                #[doc = concat!("assert_eq!(", stringify!($Ty), "::<", stringify!($Int), ">::BITS, ", stringify!($Int), "::BITS);")]
                /// ```
                #[stable(feature = "nonzero_bits", since = "1.67.0")]
                pub const BITS: u32 = <$Int>::BITS;
            }
        )+
    }
}

nonzero_bits! {
    NonZero<u8>;
    NonZero<i8>;
    NonZero<u16>;
    NonZero<i16>;
    NonZero<u32>;
    NonZero<i32>;
    NonZero<u64>;
    NonZero<i64>;
    NonZero<u128>;
    NonZero<i128>;
    NonZero<usize>;
    NonZero<isize>;
}
