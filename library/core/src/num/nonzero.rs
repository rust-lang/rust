//! Definitions of integer that is known not to equal zero.

use super::{IntErrorKind, ParseIntError};
use crate::cmp::Ordering;
use crate::hash::{Hash, Hasher};
use crate::marker::{Freeze, StructuralPartialEq};
use crate::ops::{BitOr, BitOrAssign, Div, DivAssign, Neg, Rem, RemAssign};
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::str::FromStr;
use crate::{fmt, intrinsics, ptr, ub_checks};

/// A marker trait for primitive types which can be zero.
///
/// This is an implementation detail for <code>[NonZero]\<T></code> which may disappear or be replaced at any time.
///
/// # Safety
///
/// Types implementing this trait must be primitives that are valid when zeroed.
///
/// The associated `Self::NonZeroInner` type must have the same size+align as `Self`,
/// but with a niche and bit validity making it so the following `transmutes` are sound:
///
/// - `Self::NonZeroInner` to `Option<Self::NonZeroInner>`
/// - `Option<Self::NonZeroInner>` to `Self`
///
/// (And, consequently, `Self::NonZeroInner` to `Self`.)
#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
pub unsafe trait ZeroablePrimitive: Sized + Copy + private::Sealed {
    #[doc(hidden)]
    type NonZeroInner: Sized + Copy;
}

macro_rules! impl_zeroable_primitive {
    ($($NonZeroInner:ident ( $primitive:ty )),+ $(,)?) => {
        mod private {
            #[unstable(
                feature = "nonzero_internals",
                reason = "implementation detail which may disappear or be replaced at any time",
                issue = "none"
            )]
            pub trait Sealed {}
        }

        $(
            #[unstable(
                feature = "nonzero_internals",
                reason = "implementation detail which may disappear or be replaced at any time",
                issue = "none"
            )]
            impl private::Sealed for $primitive {}

            #[unstable(
                feature = "nonzero_internals",
                reason = "implementation detail which may disappear or be replaced at any time",
                issue = "none"
            )]
            unsafe impl ZeroablePrimitive for $primitive {
                type NonZeroInner = super::niche_types::$NonZeroInner;
            }
        )+
    };
}

impl_zeroable_primitive!(
    NonZeroU8Inner(u8),
    NonZeroU16Inner(u16),
    NonZeroU32Inner(u32),
    NonZeroU64Inner(u64),
    NonZeroU128Inner(u128),
    NonZeroUsizeInner(usize),
    NonZeroI8Inner(i8),
    NonZeroI16Inner(i16),
    NonZeroI32Inner(i32),
    NonZeroI64Inner(i64),
    NonZeroI128Inner(i128),
    NonZeroIsizeInner(isize),
);

/// A value that is known not to equal zero.
///
/// This enables some memory layout optimization.
/// For example, `Option<NonZero<u32>>` is the same size as `u32`:
///
/// ```
/// use core::{mem::size_of, num::NonZero};
///
/// assert_eq!(size_of::<Option<NonZero<u32>>>(), size_of::<u32>());
/// ```
///
/// # Layout
///
/// `NonZero<T>` is guaranteed to have the same layout and bit validity as `T`
/// with the exception that the all-zero bit pattern is invalid.
/// `Option<NonZero<T>>` is guaranteed to be compatible with `T`, including in
/// FFI.
///
/// Thanks to the [null pointer optimization], `NonZero<T>` and
/// `Option<NonZero<T>>` are guaranteed to have the same size and alignment:
///
/// ```
/// # use std::mem::{size_of, align_of};
/// use std::num::NonZero;
///
/// assert_eq!(size_of::<NonZero<u32>>(), size_of::<Option<NonZero<u32>>>());
/// assert_eq!(align_of::<NonZero<u32>>(), align_of::<Option<NonZero<u32>>>());
/// ```
///
/// [null pointer optimization]: crate::option#representation
#[stable(feature = "generic_nonzero", since = "1.79.0")]
#[repr(transparent)]
#[rustc_nonnull_optimization_guaranteed]
#[rustc_diagnostic_item = "NonZero"]
pub struct NonZero<T: ZeroablePrimitive>(T::NonZeroInner);

macro_rules! impl_nonzero_fmt {
    ($(#[$Attribute:meta] $Trait:ident)*) => {
        $(
            #[$Attribute]
            impl<T> fmt::$Trait for NonZero<T>
            where
                T: ZeroablePrimitive + fmt::$Trait,
            {
                #[inline]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.get().fmt(f)
                }
            }
        )*
    };
}

impl_nonzero_fmt! {
    #[stable(feature = "nonzero", since = "1.28.0")]
    Debug
    #[stable(feature = "nonzero", since = "1.28.0")]
    Display
    #[stable(feature = "nonzero", since = "1.28.0")]
    Binary
    #[stable(feature = "nonzero", since = "1.28.0")]
    Octal
    #[stable(feature = "nonzero", since = "1.28.0")]
    LowerHex
    #[stable(feature = "nonzero", since = "1.28.0")]
    UpperHex
    #[stable(feature = "nonzero_fmt_exp", since = "1.84.0")]
    LowerExp
    #[stable(feature = "nonzero_fmt_exp", since = "1.84.0")]
    UpperExp
}

macro_rules! impl_nonzero_auto_trait {
    (unsafe $Trait:ident) => {
        #[stable(feature = "nonzero", since = "1.28.0")]
        unsafe impl<T> $Trait for NonZero<T> where T: ZeroablePrimitive + $Trait {}
    };
    ($Trait:ident) => {
        #[stable(feature = "nonzero", since = "1.28.0")]
        impl<T> $Trait for NonZero<T> where T: ZeroablePrimitive + $Trait {}
    };
}

// Implement auto-traits manually based on `T` to avoid docs exposing
// the `ZeroablePrimitive::NonZeroInner` implementation detail.
impl_nonzero_auto_trait!(unsafe Freeze);
impl_nonzero_auto_trait!(RefUnwindSafe);
impl_nonzero_auto_trait!(unsafe Send);
impl_nonzero_auto_trait!(unsafe Sync);
impl_nonzero_auto_trait!(Unpin);
impl_nonzero_auto_trait!(UnwindSafe);

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> Clone for NonZero<T>
where
    T: ZeroablePrimitive,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> Copy for NonZero<T> where T: ZeroablePrimitive {}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> PartialEq for NonZero<T>
where
    T: ZeroablePrimitive + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        self.get() != other.get()
    }
}

#[unstable(feature = "structural_match", issue = "31434")]
impl<T> StructuralPartialEq for NonZero<T> where T: ZeroablePrimitive + StructuralPartialEq {}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> Eq for NonZero<T> where T: ZeroablePrimitive + Eq {}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> PartialOrd for NonZero<T>
where
    T: ZeroablePrimitive + PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.get() < other.get()
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.get() <= other.get()
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.get() > other.get()
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.get() >= other.get()
    }
}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> Ord for NonZero<T>
where
    T: ZeroablePrimitive + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(&other.get())
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        // SAFETY: The maximum of two non-zero values is still non-zero.
        unsafe { Self::new_unchecked(self.get().max(other.get())) }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        // SAFETY: The minimum of two non-zero values is still non-zero.
        unsafe { Self::new_unchecked(self.get().min(other.get())) }
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        // SAFETY: A non-zero value clamped between two non-zero values is still non-zero.
        unsafe { Self::new_unchecked(self.get().clamp(min.get(), max.get())) }
    }
}

#[stable(feature = "nonzero", since = "1.28.0")]
impl<T> Hash for NonZero<T>
where
    T: ZeroablePrimitive + Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.get().hash(state)
    }
}

#[stable(feature = "from_nonzero", since = "1.31.0")]
impl<T> From<NonZero<T>> for T
where
    T: ZeroablePrimitive,
{
    #[inline]
    fn from(nonzero: NonZero<T>) -> Self {
        // Call `get` method to keep range information.
        nonzero.get()
    }
}

#[stable(feature = "nonzero_bitor", since = "1.45.0")]
impl<T> BitOr for NonZero<T>
where
    T: ZeroablePrimitive + BitOr<Output = T>,
{
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        // SAFETY: Bitwise OR of two non-zero values is still non-zero.
        unsafe { Self::new_unchecked(self.get() | rhs.get()) }
    }
}

#[stable(feature = "nonzero_bitor", since = "1.45.0")]
impl<T> BitOr<T> for NonZero<T>
where
    T: ZeroablePrimitive + BitOr<Output = T>,
{
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: T) -> Self::Output {
        // SAFETY: Bitwise OR of a non-zero value with anything is still non-zero.
        unsafe { Self::new_unchecked(self.get() | rhs) }
    }
}

#[stable(feature = "nonzero_bitor", since = "1.45.0")]
impl<T> BitOr<NonZero<T>> for T
where
    T: ZeroablePrimitive + BitOr<Output = T>,
{
    type Output = NonZero<T>;

    #[inline]
    fn bitor(self, rhs: NonZero<T>) -> Self::Output {
        // SAFETY: Bitwise OR of anything with a non-zero value is still non-zero.
        unsafe { NonZero::new_unchecked(self | rhs.get()) }
    }
}

#[stable(feature = "nonzero_bitor", since = "1.45.0")]
impl<T> BitOrAssign for NonZero<T>
where
    T: ZeroablePrimitive,
    Self: BitOr<Output = Self>,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[stable(feature = "nonzero_bitor", since = "1.45.0")]
impl<T> BitOrAssign<T> for NonZero<T>
where
    T: ZeroablePrimitive,
    Self: BitOr<T, Output = Self>,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: T) {
        *self = *self | rhs;
    }
}

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
    /// This results in undefined behavior if the value is zero.
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
                    ub_checks::assert_unsafe_precondition!(
                        check_language_ub,
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
        let opt_n = unsafe { &mut *(ptr::from_mut(n).cast::<Option<Self>>()) };

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
                    ub_checks::assert_unsafe_precondition!(
                        check_library_ub,
                        "NonZero::from_mut_unchecked requires the argument to dereference as non-zero",
                        () => false,
                    );
                    intrinsics::unreachable()
                }
            }
        }
    }

    /// Returns the contained value as a primitive type.
    #[stable(feature = "nonzero", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_nonzero_get", since = "1.34.0")]
    #[inline]
    pub const fn get(self) -> T {
        // Rustc can set range metadata only if it loads `self` from
        // memory somewhere. If the value of `self` was from by-value argument
        // of some not-inlined function, LLVM don't have range metadata
        // to understand that the value cannot be zero.
        //
        // Using the transmute `assume`s the range at runtime.
        //
        // Even once LLVM supports `!range` metadata for function arguments
        // (see <https://github.com/llvm/llvm-project/issues/76628>), this can't
        // be `.0` because MCP#807 bans field-projecting into `scalar_valid_range`
        // types, and it arguably wouldn't want to be anyway because if this is
        // MIR-inlined, there's no opportunity to put that argument metadata anywhere.
        //
        // The good answer here will eventually be pattern types, which will hopefully
        // allow it to go back to `.0`, maybe with a cast of some sort.
        //
        // SAFETY: `ZeroablePrimitive` guarantees that the size and bit validity
        // of `.0` is such that this transmute is sound.
        unsafe { intrinsics::transmute_unchecked(self) }
    }
}

macro_rules! nonzero_integer {
    (
        #[$stability:meta]
        Self = $Ty:ident,
        Primitive = $signedness:ident $Int:ident,
        SignedPrimitive = $Sint:ty,
        UnsignedPrimitive = $Uint:ty,

        // Used in doc comments.
        rot = $rot:literal,
        rot_op = $rot_op:literal,
        rot_result = $rot_result:literal,
        swap_op = $swap_op:literal,
        swapped = $swapped:literal,
        reversed = $reversed:literal,
        leading_zeros_test = $leading_zeros_test:expr,
    ) => {
        #[doc = sign_dependent_expr!{
            $signedness ?
            if signed {
                concat!("An [`", stringify!($Int), "`] that is known not to equal zero.")
            }
            if unsigned {
                concat!("A [`", stringify!($Int), "`] that is known not to equal zero.")
            }
        }]
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

        impl NonZero<$Int> {
            /// The size of this non-zero integer type in bits.
            ///
            #[doc = concat!("This value is equal to [`", stringify!($Int), "::BITS`].")]
            ///
            /// # Examples
            ///
            /// ```
            /// # use std::num::NonZero;
            /// #
            #[doc = concat!("assert_eq!(NonZero::<", stringify!($Int), ">::BITS, ", stringify!($Int), "::BITS);")]
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
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::<", stringify!($Int), ">::new(", $leading_zeros_test, ")?;")]
            ///
            /// assert_eq!(n.leading_zeros(), 0);
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[rustc_const_stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn leading_zeros(self) -> u32 {
                // SAFETY: since `self` cannot be zero, it is safe to call `ctlz_nonzero`.
                unsafe {
                    intrinsics::ctlz_nonzero(self.get() as $Uint)
                }
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
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::<", stringify!($Int), ">::new(0b0101000)?;")]
            ///
            /// assert_eq!(n.trailing_zeros(), 3);
            /// # Some(())
            /// # }
            /// ```
            #[stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[rustc_const_stable(feature = "nonzero_leading_trailing_zeros", since = "1.53.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn trailing_zeros(self) -> u32 {
                // SAFETY: since `self` cannot be zero, it is safe to call `cttz_nonzero`.
                unsafe {
                    intrinsics::cttz_nonzero(self.get() as $Uint)
                }
            }

            /// Returns `self` with only the most significant bit set.
            ///
            /// # Example
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(isolate_most_least_significant_one)]
            ///
            /// # use core::num::NonZero;
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let a = NonZero::<", stringify!($Int), ">::new(0b_01100100)?;")]
            #[doc = concat!("let b = NonZero::<", stringify!($Int), ">::new(0b_01000000)?;")]
            ///
            /// assert_eq!(a.isolate_most_significant_one(), b);
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "isolate_most_least_significant_one", issue = "136909")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn isolate_most_significant_one(self) -> Self {
                let n = self.get() & (((1 as $Int) << (<$Int>::BITS - 1)).wrapping_shr(self.leading_zeros()));

                // SAFETY:
                // `self` is non-zero, so masking to preserve only the most
                // significant set bit will result in a non-zero `n`.
                unsafe { NonZero::new_unchecked(n) }
            }

            /// Returns `self` with only the least significant bit set.
            ///
            /// # Example
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(isolate_most_least_significant_one)]
            ///
            /// # use core::num::NonZero;
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let a = NonZero::<", stringify!($Int), ">::new(0b_01100100)?;")]
            #[doc = concat!("let b = NonZero::<", stringify!($Int), ">::new(0b_00000100)?;")]
            ///
            /// assert_eq!(a.isolate_least_significant_one(), b);
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "isolate_most_least_significant_one", issue = "136909")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn isolate_least_significant_one(self) -> Self {
                let n = self.get();
                let n = n & n.wrapping_neg();

                // SAFETY: `self` is non-zero, so `self` with only its least
                // significant set bit will remain non-zero.
                unsafe { NonZero::new_unchecked(n) }
            }

            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(non_zero_count_ones)]
            ///
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let a = NonZero::<", stringify!($Int), ">::new(0b100_0000)?;")]
            #[doc = concat!("let b = NonZero::<", stringify!($Int), ">::new(0b100_0011)?;")]
            ///
            /// assert_eq!(a.count_ones(), NonZero::new(1)?);
            /// assert_eq!(b.count_ones(), NonZero::new(3)?);
            /// # Some(())
            /// # }
            /// ```
            ///
            #[unstable(feature = "non_zero_count_ones", issue = "120287")]
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn count_ones(self) -> NonZero<u32> {
                // SAFETY:
                // `self` is non-zero, which means it has at least one bit set, which means
                // that the result of `count_ones` is non-zero.
                unsafe { NonZero::new_unchecked(self.get().count_ones()) }
            }

            /// Shifts the bits to the left by a specified amount, `n`,
            /// wrapping the truncated bits to the end of the resulting integer.
            ///
            /// Please note this isn't the same operation as the `<<` shifting operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(", $rot_op, stringify!($Int), ")?;")]
            #[doc = concat!("let m = NonZero::new(", $rot_result, ")?;")]
            ///
            #[doc = concat!("assert_eq!(n.rotate_left(", $rot, "), m);")]
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn rotate_left(self, n: u32) -> Self {
                let result = self.get().rotate_left(n);
                // SAFETY: Rotating bits preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Shifts the bits to the right by a specified amount, `n`,
            /// wrapping the truncated bits to the beginning of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as the `>>` shifting operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(", $rot_result, stringify!($Int), ")?;")]
            #[doc = concat!("let m = NonZero::new(", $rot_op, ")?;")]
            ///
            #[doc = concat!("assert_eq!(n.rotate_right(", $rot, "), m);")]
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn rotate_right(self, n: u32) -> Self {
                let result = self.get().rotate_right(n);
                // SAFETY: Rotating bits preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Reverses the byte order of the integer.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(", $swap_op, stringify!($Int), ")?;")]
            /// let m = n.swap_bytes();
            ///
            #[doc = concat!("assert_eq!(m, NonZero::new(", $swapped, ")?);")]
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn swap_bytes(self) -> Self {
                let result = self.get().swap_bytes();
                // SAFETY: Shuffling bytes preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Reverses the order of bits in the integer. The least significant bit becomes the most significant bit,
            /// second least-significant bit becomes second most-significant bit, etc.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(", $swap_op, stringify!($Int), ")?;")]
            /// let m = n.reverse_bits();
            ///
            #[doc = concat!("assert_eq!(m, NonZero::new(", $reversed, ")?);")]
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn reverse_bits(self) -> Self {
                let result = self.get().reverse_bits();
                // SAFETY: Reversing bits preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Converts an integer from big endian to the target's endianness.
            ///
            /// On big endian this is a no-op. On little endian the bytes are
            /// swapped.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            #[doc = concat!("use std::num::", stringify!($Ty), ";")]
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(0x1A", stringify!($Int), ")?;")]
            ///
            /// if cfg!(target_endian = "big") {
            #[doc = concat!("    assert_eq!(", stringify!($Ty), "::from_be(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(", stringify!($Ty), "::from_be(n), n.swap_bytes())")]
            /// }
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use]
            #[inline(always)]
            pub const fn from_be(x: Self) -> Self {
                let result = $Int::from_be(x.get());
                // SAFETY: Shuffling bytes preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Converts an integer from little endian to the target's endianness.
            ///
            /// On little endian this is a no-op. On big endian the bytes are
            /// swapped.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            #[doc = concat!("use std::num::", stringify!($Ty), ";")]
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(0x1A", stringify!($Int), ")?;")]
            ///
            /// if cfg!(target_endian = "little") {
            #[doc = concat!("    assert_eq!(", stringify!($Ty), "::from_le(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(", stringify!($Ty), "::from_le(n), n.swap_bytes())")]
            /// }
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use]
            #[inline(always)]
            pub const fn from_le(x: Self) -> Self {
                let result = $Int::from_le(x.get());
                // SAFETY: Shuffling bytes preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Converts `self` to big endian from the target's endianness.
            ///
            /// On big endian this is a no-op. On little endian the bytes are
            /// swapped.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(0x1A", stringify!($Int), ")?;")]
            ///
            /// if cfg!(target_endian = "big") {
            ///     assert_eq!(n.to_be(), n)
            /// } else {
            ///     assert_eq!(n.to_be(), n.swap_bytes())
            /// }
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn to_be(self) -> Self {
                let result = self.get().to_be();
                // SAFETY: Shuffling bytes preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            /// Converts `self` to little endian from the target's endianness.
            ///
            /// On little endian this is a no-op. On big endian the bytes are
            /// swapped.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(nonzero_bitwise)]
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let n = NonZero::new(0x1A", stringify!($Int), ")?;")]
            ///
            /// if cfg!(target_endian = "little") {
            ///     assert_eq!(n.to_le(), n)
            /// } else {
            ///     assert_eq!(n.to_le(), n.swap_bytes())
            /// }
            /// # Some(())
            /// # }
            /// ```
            #[unstable(feature = "nonzero_bitwise", issue = "128281")]
            #[must_use = "this returns the result of the operation, \
                        without modifying the original"]
            #[inline(always)]
            pub const fn to_le(self) -> Self {
                let result = self.get().to_le();
                // SAFETY: Shuffling bytes preserves the property int > 0.
                unsafe { Self::new_unchecked(result) }
            }

            nonzero_integer_signedness_dependent_methods! {
                Primitive = $signedness $Int,
                SignedPrimitive = $Sint,
                UnsignedPrimitive = $Uint,
            }

            /// Multiplies two non-zero integers together.
            /// Checks for overflow and returns [`None`] on overflow.
            /// As a consequence, the result cannot wrap to zero.
            ///
            /// # Examples
            ///
            /// ```
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
            #[doc = concat!("let four = NonZero::new(4", stringify!($Int), ")?;")]
            #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
            #[doc = concat!("Return [`NonZero::<", stringify!($Int), ">::MAX`] on overflow.")]
            ///
            /// # Examples
            ///
            /// ```
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
            #[doc = concat!("let four = NonZero::new(4", stringify!($Int), ")?;")]
            #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
            /// Overflow is unchecked, and it is undefined behavior to overflow
            /// *even if the result would wrap to a non-zero value*.
            /// The behavior is undefined as soon as
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
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
            #[doc = concat!("let four = NonZero::new(4", stringify!($Int), ")?;")]
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
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let three = NonZero::new(3", stringify!($Int), ")?;")]
            #[doc = concat!("let twenty_seven = NonZero::new(27", stringify!($Int), ")?;")]
            #[doc = concat!("let half_max = NonZero::new(", stringify!($Int), "::MAX / 2)?;")]
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
                    concat!("Return [`NonZero::<", stringify!($Int), ">::MIN`] ",
                                "or [`NonZero::<", stringify!($Int), ">::MAX`] on overflow.")
                }
                if unsigned {
                    concat!("Return [`NonZero::<", stringify!($Int), ">::MAX`] on overflow.")
                }
            }]
            ///
            /// # Examples
            ///
            /// ```
            /// # use std::num::NonZero;
            /// #
            /// # fn main() { test().unwrap(); }
            /// # fn test() -> Option<()> {
            #[doc = concat!("let three = NonZero::new(3", stringify!($Int), ")?;")]
            #[doc = concat!("let twenty_seven = NonZero::new(27", stringify!($Int), ")?;")]
            #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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

        #[stable(feature = "nonzero_parse", since = "1.35.0")]
        impl FromStr for NonZero<$Int> {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, Self::Err> {
                Self::new(<$Int>::from_str_radix(src, 10)?)
                    .ok_or(ParseIntError {
                        kind: IntErrorKind::Zero
                    })
            }
        }

        nonzero_integer_signedness_dependent_impls!($signedness $Int);
    };

    (
        Self = $Ty:ident,
        Primitive = unsigned $Int:ident,
        SignedPrimitive = $Sint:ident,
        rot = $rot:literal,
        rot_op = $rot_op:literal,
        rot_result = $rot_result:literal,
        swap_op = $swap_op:literal,
        swapped = $swapped:literal,
        reversed = $reversed:literal,
        $(,)?
    ) => {
        nonzero_integer! {
            #[stable(feature = "nonzero", since = "1.28.0")]
            Self = $Ty,
            Primitive = unsigned $Int,
            SignedPrimitive = $Sint,
            UnsignedPrimitive = $Int,
            rot = $rot,
            rot_op = $rot_op,
            rot_result = $rot_result,
            swap_op = $swap_op,
            swapped = $swapped,
            reversed = $reversed,
            leading_zeros_test = concat!(stringify!($Int), "::MAX"),
        }
    };

    (
        Self = $Ty:ident,
        Primitive = signed $Int:ident,
        UnsignedPrimitive = $Uint:ident,
        rot = $rot:literal,
        rot_op = $rot_op:literal,
        rot_result = $rot_result:literal,
        swap_op = $swap_op:literal,
        swapped = $swapped:literal,
        reversed = $reversed:literal,
    ) => {
        nonzero_integer! {
            #[stable(feature = "signed_nonzero", since = "1.34.0")]
            Self = $Ty,
            Primitive = signed $Int,
            SignedPrimitive = $Int,
            UnsignedPrimitive = $Uint,
            rot = $rot,
            rot_op = $rot_op,
            rot_result = $rot_result,
            swap_op = $swap_op,
            swapped = $swapped,
            reversed = $reversed,
            leading_zeros_test = concat!("-1", stringify!($Int)),
        }
    };
}

macro_rules! nonzero_integer_signedness_dependent_impls {
    // Impls for unsigned nonzero types only.
    (unsigned $Int:ty) => {
        #[stable(feature = "nonzero_div", since = "1.51.0")]
        impl Div<NonZero<$Int>> for $Int {
            type Output = $Int;

            /// Same as `self / other.get()`, but because `other` is a `NonZero<_>`,
            /// there's never a runtime check for division-by-zero.
            ///
            /// This operation rounds towards zero, truncating any fractional
            /// part of the exact result, and cannot panic.
            #[doc(alias = "unchecked_div")]
            #[inline]
            fn div(self, other: NonZero<$Int>) -> $Int {
                // SAFETY: Division by zero is checked because `other` is non-zero,
                // and MIN/-1 is checked because `self` is an unsigned int.
                unsafe { intrinsics::unchecked_div(self, other.get()) }
            }
        }

        #[stable(feature = "nonzero_div_assign", since = "1.79.0")]
        impl DivAssign<NonZero<$Int>> for $Int {
            /// Same as `self /= other.get()`, but because `other` is a `NonZero<_>`,
            /// there's never a runtime check for division-by-zero.
            ///
            /// This operation rounds towards zero, truncating any fractional
            /// part of the exact result, and cannot panic.
            #[inline]
            fn div_assign(&mut self, other: NonZero<$Int>) {
                *self = *self / other;
            }
        }

        #[stable(feature = "nonzero_div", since = "1.51.0")]
        impl Rem<NonZero<$Int>> for $Int {
            type Output = $Int;

            /// This operation satisfies `n % d == n - (n / d) * d`, and cannot panic.
            #[inline]
            fn rem(self, other: NonZero<$Int>) -> $Int {
                // SAFETY: Remainder by zero is checked because `other` is non-zero,
                // and MIN/-1 is checked because `self` is an unsigned int.
                unsafe { intrinsics::unchecked_rem(self, other.get()) }
            }
        }

        #[stable(feature = "nonzero_div_assign", since = "1.79.0")]
        impl RemAssign<NonZero<$Int>> for $Int {
            /// This operation satisfies `n % d == n - (n / d) * d`, and cannot panic.
            #[inline]
            fn rem_assign(&mut self, other: NonZero<$Int>) {
                *self = *self % other;
            }
        }

        impl NonZero<$Int> {
            /// Calculates the quotient of `self` and `rhs`, rounding the result towards positive infinity.
            ///
            /// The result is guaranteed to be non-zero.
            ///
            /// # Examples
            ///
            /// ```
            /// # #![feature(unsigned_nonzero_div_ceil)]
            /// # use std::num::NonZero;
            #[doc = concat!("let one = NonZero::new(1", stringify!($Int), ").unwrap();")]
            #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX).unwrap();")]
            /// assert_eq!(one.div_ceil(max), one);
            ///
            #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ").unwrap();")]
            #[doc = concat!("let three = NonZero::new(3", stringify!($Int), ").unwrap();")]
            /// assert_eq!(three.div_ceil(two), two);
            /// ```
            #[unstable(feature = "unsigned_nonzero_div_ceil", issue = "132968")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn div_ceil(self, rhs: Self) -> Self {
                let v = self.get().div_ceil(rhs.get());
                // SAFETY: ceiled division of two positive integers can never be zero.
                unsafe { Self::new_unchecked(v) }
            }
        }
    };
    // Impls for signed nonzero types only.
    (signed $Int:ty) => {
        #[stable(feature = "signed_nonzero_neg", since = "1.71.0")]
        impl Neg for NonZero<$Int> {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                // SAFETY: negation of nonzero cannot yield zero values.
                unsafe { Self::new_unchecked(self.get().neg()) }
            }
        }

        forward_ref_unop! { impl Neg, neg for NonZero<$Int>,
        #[stable(feature = "signed_nonzero_neg", since = "1.71.0")] }
    };
}

#[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5974
macro_rules! nonzero_integer_signedness_dependent_methods {
    // Associated items for unsigned nonzero types only.
    (
        Primitive = unsigned $Int:ident,
        SignedPrimitive = $Sint:ty,
        UnsignedPrimitive = $Uint:ty,
    ) => {
        /// The smallest value that can be represented by this non-zero
        /// integer type, 1.
        ///
        /// # Examples
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        #[doc = concat!("assert_eq!(NonZero::<", stringify!($Int), ">::MIN.get(), 1", stringify!($Int), ");")]
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
        /// # use std::num::NonZero;
        /// #
        #[doc = concat!("assert_eq!(NonZero::<", stringify!($Int), ">::MAX.get(), ", stringify!($Int), "::MAX);")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
        #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        #[doc = concat!("Return [`NonZero::<", stringify!($Int), ">::MAX`] on overflow.")]
        ///
        /// # Examples
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
        #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        /// Overflow is unchecked, and it is undefined behavior to overflow
        /// *even if the result would wrap to a non-zero value*.
        /// The behavior is undefined as soon as
        #[doc = concat!("`self + rhs > ", stringify!($Int), "::MAX`.")]
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(nonzero_ops)]
        ///
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
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

        /// Returns the smallest power of two greater than or equal to `self`.
        /// Checks for overflow and returns [`None`]
        /// if the next power of two is greater than the type’s maximum value.
        /// As a consequence, the result cannot wrap to zero.
        ///
        /// # Examples
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
        #[doc = concat!("let three = NonZero::new(3", stringify!($Int), ")?;")]
        #[doc = concat!("let four = NonZero::new(4", stringify!($Int), ")?;")]
        #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("assert_eq!(NonZero::new(7", stringify!($Int), ")?.ilog2(), 2);")]
        #[doc = concat!("assert_eq!(NonZero::new(8", stringify!($Int), ")?.ilog2(), 3);")]
        #[doc = concat!("assert_eq!(NonZero::new(9", stringify!($Int), ")?.ilog2(), 3);")]
        /// # Some(())
        /// # }
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("assert_eq!(NonZero::new(99", stringify!($Int), ")?.ilog10(), 1);")]
        #[doc = concat!("assert_eq!(NonZero::new(100", stringify!($Int), ")?.ilog10(), 2);")]
        #[doc = concat!("assert_eq!(NonZero::new(101", stringify!($Int), ")?.ilog10(), 2);")]
        /// # Some(())
        /// # }
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let one = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let two = NonZero::new(2", stringify!($Int), ")?;")]
        #[doc = concat!("let four = NonZero::new(4", stringify!($Int), ")?;")]
        ///
        /// assert_eq!(one.midpoint(four), two);
        /// assert_eq!(four.midpoint(one), two);
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "num_midpoint", since = "1.85.0")]
        #[rustc_const_stable(feature = "num_midpoint", since = "1.85.0")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let eight = NonZero::new(8", stringify!($Int), ")?;")]
        /// assert!(eight.is_power_of_two());
        #[doc = concat!("let ten = NonZero::new(10", stringify!($Int), ")?;")]
        /// assert!(!ten.is_power_of_two());
        /// # Some(())
        /// # }
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

        /// Returns the square root of the number, rounded down.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let ten = NonZero::new(10", stringify!($Int), ")?;")]
        #[doc = concat!("let three = NonZero::new(3", stringify!($Int), ")?;")]
        ///
        /// assert_eq!(ten.isqrt(), three);
        /// # Some(())
        /// # }
        /// ```
        #[stable(feature = "isqrt", since = "1.84.0")]
        #[rustc_const_stable(feature = "isqrt", since = "1.84.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn isqrt(self) -> Self {
            let result = self.get().isqrt();

            // SAFETY: Integer square root is a monotonically nondecreasing
            // function, which means that increasing the input will never cause
            // the output to decrease. Thus, since the input for nonzero
            // unsigned integers has a lower bound of 1, the lower bound of the
            // results will be sqrt(1), which is 1, so a result can't be zero.
            unsafe { Self::new_unchecked(result) }
        }

        /// Returns the bit pattern of `self` reinterpreted as a signed integer of the same size.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(integer_sign_cast)]
        /// # use std::num::NonZero;
        ///
        #[doc = concat!("let n = NonZero::<", stringify!($Int), ">::MAX;")]
        ///
        #[doc = concat!("assert_eq!(n.cast_signed(), NonZero::new(-1", stringify!($Sint), ").unwrap());")]
        /// ```
        #[unstable(feature = "integer_sign_cast", issue = "125882")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn cast_signed(self) -> NonZero<$Sint> {
            // SAFETY: `self.get()` can't be zero
            unsafe { NonZero::new_unchecked(self.get().cast_signed()) }
        }
    };

    // Associated items for signed nonzero types only.
    (
        Primitive = signed $Int:ident,
        SignedPrimitive = $Sint:ty,
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
        /// # use std::num::NonZero;
        /// #
        #[doc = concat!("assert_eq!(NonZero::<", stringify!($Int), ">::MIN.get(), ", stringify!($Int), "::MIN);")]
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
        /// # use std::num::NonZero;
        /// #
        #[doc = concat!("assert_eq!(NonZero::<", stringify!($Int), ">::MAX.get(), ", stringify!($Int), "::MAX);")]
        /// ```
        #[stable(feature = "nonzero_min_max", since = "1.70.0")]
        pub const MAX: Self = Self::new(<$Int>::MAX).unwrap();

        /// Computes the absolute value of self.
        #[doc = concat!("See [`", stringify!($Int), "::abs`]")]
        /// for documentation on overflow behavior.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let neg = NonZero::new(-1", stringify!($Int), ")?;")]
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
        #[doc = concat!("`self == NonZero::<", stringify!($Int), ">::MIN`.")]
        /// The result cannot be zero.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let neg = NonZero::new(-1", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let neg = NonZero::new(-1", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let neg = NonZero::new(-1", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let min_plus = NonZero::new(", stringify!($Int), "::MIN + 1)?;")]
        #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let neg = NonZero::new(-1", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
        #[doc = concat!("# let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let u_pos = NonZero::new(1", stringify!($Uint), ")?;")]
        #[doc = concat!("let i_pos = NonZero::new(1", stringify!($Int), ")?;")]
        #[doc = concat!("let i_neg = NonZero::new(-1", stringify!($Int), ")?;")]
        #[doc = concat!("let i_min = NonZero::new(", stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let u_max = NonZero::new(", stringify!($Uint), "::MAX / 2 + 1)?;")]
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
        pub const fn unsigned_abs(self) -> NonZero<$Uint> {
            // SAFETY: absolute value of nonzero cannot yield zero values.
            unsafe { NonZero::new_unchecked(self.get().unsigned_abs()) }
        }

        /// Returns `true` if `self` is positive and `false` if the
        /// number is negative.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
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
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
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
        #[doc = concat!("returning `None` if `self == NonZero::<", stringify!($Int), ">::MIN`.")]
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
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
        /// for documentation on overflow behavior.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
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
        #[doc = concat!("returning [`NonZero::<", stringify!($Int), ">::MAX`]")]
        #[doc = concat!("if `self == NonZero::<", stringify!($Int), ">::MIN`")]
        /// instead of overflowing.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
        #[doc = concat!("let min_plus_one = NonZero::new(", stringify!($Int), "::MIN + 1)?;")]
        #[doc = concat!("let max = NonZero::new(", stringify!($Int), "::MAX)?;")]
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
        /// for documentation on overflow behavior.
        ///
        /// # Example
        ///
        /// ```
        /// # use std::num::NonZero;
        /// #
        /// # fn main() { test().unwrap(); }
        /// # fn test() -> Option<()> {
        #[doc = concat!("let pos_five = NonZero::new(5", stringify!($Int), ")?;")]
        #[doc = concat!("let neg_five = NonZero::new(-5", stringify!($Int), ")?;")]
        #[doc = concat!("let min = NonZero::new(", stringify!($Int), "::MIN)?;")]
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

        /// Returns the bit pattern of `self` reinterpreted as an unsigned integer of the same size.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(integer_sign_cast)]
        /// # use std::num::NonZero;
        ///
        #[doc = concat!("let n = NonZero::new(-1", stringify!($Int), ").unwrap();")]
        ///
        #[doc = concat!("assert_eq!(n.cast_unsigned(), NonZero::<", stringify!($Uint), ">::MAX);")]
        /// ```
        #[unstable(feature = "integer_sign_cast", issue = "125882")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn cast_unsigned(self) -> NonZero<$Uint> {
            // SAFETY: `self.get()` can't be zero
            unsafe { NonZero::new_unchecked(self.get().cast_unsigned()) }
        }

    };
}

nonzero_integer! {
    Self = NonZeroU8,
    Primitive = unsigned u8,
    SignedPrimitive = i8,
    rot = 2,
    rot_op = "0x82",
    rot_result = "0xa",
    swap_op = "0x12",
    swapped = "0x12",
    reversed = "0x48",
}

nonzero_integer! {
    Self = NonZeroU16,
    Primitive = unsigned u16,
    SignedPrimitive = i16,
    rot = 4,
    rot_op = "0xa003",
    rot_result = "0x3a",
    swap_op = "0x1234",
    swapped = "0x3412",
    reversed = "0x2c48",
}

nonzero_integer! {
    Self = NonZeroU32,
    Primitive = unsigned u32,
    SignedPrimitive = i32,
    rot = 8,
    rot_op = "0x10000b3",
    rot_result = "0xb301",
    swap_op = "0x12345678",
    swapped = "0x78563412",
    reversed = "0x1e6a2c48",
}

nonzero_integer! {
    Self = NonZeroU64,
    Primitive = unsigned u64,
    SignedPrimitive = i64,
    rot = 12,
    rot_op = "0xaa00000000006e1",
    rot_result = "0x6e10aa",
    swap_op = "0x1234567890123456",
    swapped = "0x5634129078563412",
    reversed = "0x6a2c48091e6a2c48",
}

nonzero_integer! {
    Self = NonZeroU128,
    Primitive = unsigned u128,
    SignedPrimitive = i128,
    rot = 16,
    rot_op = "0x13f40000000000000000000000004f76",
    rot_result = "0x4f7613f4",
    swap_op = "0x12345678901234567890123456789012",
    swapped = "0x12907856341290785634129078563412",
    reversed = "0x48091e6a2c48091e6a2c48091e6a2c48",
}

#[cfg(target_pointer_width = "16")]
nonzero_integer! {
    Self = NonZeroUsize,
    Primitive = unsigned usize,
    SignedPrimitive = isize,
    rot = 4,
    rot_op = "0xa003",
    rot_result = "0x3a",
    swap_op = "0x1234",
    swapped = "0x3412",
    reversed = "0x2c48",
}

#[cfg(target_pointer_width = "32")]
nonzero_integer! {
    Self = NonZeroUsize,
    Primitive = unsigned usize,
    SignedPrimitive = isize,
    rot = 8,
    rot_op = "0x10000b3",
    rot_result = "0xb301",
    swap_op = "0x12345678",
    swapped = "0x78563412",
    reversed = "0x1e6a2c48",
}

#[cfg(target_pointer_width = "64")]
nonzero_integer! {
    Self = NonZeroUsize,
    Primitive = unsigned usize,
    SignedPrimitive = isize,
    rot = 12,
    rot_op = "0xaa00000000006e1",
    rot_result = "0x6e10aa",
    swap_op = "0x1234567890123456",
    swapped = "0x5634129078563412",
    reversed = "0x6a2c48091e6a2c48",
}

nonzero_integer! {
    Self = NonZeroI8,
    Primitive = signed i8,
    UnsignedPrimitive = u8,
    rot = 2,
    rot_op = "-0x7e",
    rot_result = "0xa",
    swap_op = "0x12",
    swapped = "0x12",
    reversed = "0x48",
}

nonzero_integer! {
    Self = NonZeroI16,
    Primitive = signed i16,
    UnsignedPrimitive = u16,
    rot = 4,
    rot_op = "-0x5ffd",
    rot_result = "0x3a",
    swap_op = "0x1234",
    swapped = "0x3412",
    reversed = "0x2c48",
}

nonzero_integer! {
    Self = NonZeroI32,
    Primitive = signed i32,
    UnsignedPrimitive = u32,
    rot = 8,
    rot_op = "0x10000b3",
    rot_result = "0xb301",
    swap_op = "0x12345678",
    swapped = "0x78563412",
    reversed = "0x1e6a2c48",
}

nonzero_integer! {
    Self = NonZeroI64,
    Primitive = signed i64,
    UnsignedPrimitive = u64,
    rot = 12,
    rot_op = "0xaa00000000006e1",
    rot_result = "0x6e10aa",
    swap_op = "0x1234567890123456",
    swapped = "0x5634129078563412",
    reversed = "0x6a2c48091e6a2c48",
}

nonzero_integer! {
    Self = NonZeroI128,
    Primitive = signed i128,
    UnsignedPrimitive = u128,
    rot = 16,
    rot_op = "0x13f40000000000000000000000004f76",
    rot_result = "0x4f7613f4",
    swap_op = "0x12345678901234567890123456789012",
    swapped = "0x12907856341290785634129078563412",
    reversed = "0x48091e6a2c48091e6a2c48091e6a2c48",
}

#[cfg(target_pointer_width = "16")]
nonzero_integer! {
    Self = NonZeroIsize,
    Primitive = signed isize,
    UnsignedPrimitive = usize,
    rot = 4,
    rot_op = "-0x5ffd",
    rot_result = "0x3a",
    swap_op = "0x1234",
    swapped = "0x3412",
    reversed = "0x2c48",
}

#[cfg(target_pointer_width = "32")]
nonzero_integer! {
    Self = NonZeroIsize,
    Primitive = signed isize,
    UnsignedPrimitive = usize,
    rot = 8,
    rot_op = "0x10000b3",
    rot_result = "0xb301",
    swap_op = "0x12345678",
    swapped = "0x78563412",
    reversed = "0x1e6a2c48",
}

#[cfg(target_pointer_width = "64")]
nonzero_integer! {
    Self = NonZeroIsize,
    Primitive = signed isize,
    UnsignedPrimitive = usize,
    rot = 12,
    rot_op = "0xaa00000000006e1",
    rot_result = "0x6e10aa",
    swap_op = "0x1234567890123456",
    swapped = "0x5634129078563412",
    reversed = "0x6a2c48091e6a2c48",
}
