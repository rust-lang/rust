//! Definitions of integer that is known not to equal zero.

use crate::fmt;
use crate::ops::{BitOr, BitOrAssign, Div, Rem};
use crate::str::FromStr;

use super::from_str_radix;
use super::{IntErrorKind, ParseIntError};
use crate::intrinsics;

macro_rules! doc_comment {
    ($x:expr, $($tt:tt)*) => {
        #[doc = $x]
        $($tt)*
    };
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

macro_rules! nonzero_integers {
    ( $( #[$stability: meta] $Ty: ident($Int: ty); )+ ) => {
        $(
            doc_comment! {
                concat!("An integer that is known not to equal zero.

This enables some memory layout optimization.
For example, `Option<", stringify!($Ty), ">` is the same size as `", stringify!($Int), "`:

```rust
use std::mem::size_of;
assert_eq!(size_of::<Option<core::num::", stringify!($Ty), ">>(), size_of::<", stringify!($Int),
">());
```"),
                #[$stability]
                #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
                #[repr(transparent)]
                #[rustc_layout_scalar_valid_range_start(1)]
                #[rustc_nonnull_optimization_guaranteed]
                pub struct $Ty($Int);
            }

            impl $Ty {
                /// Creates a non-zero without checking the value.
                ///
                /// # Safety
                ///
                /// The value must not be zero.
                #[$stability]
                #[rustc_const_stable(feature = "nonzero", since = "1.34.0")]
                #[inline]
                pub const unsafe fn new_unchecked(n: $Int) -> Self {
                    // SAFETY: this is guaranteed to be safe by the caller.
                    unsafe { Self(n) }
                }

                /// Creates a non-zero if the given value is not zero.
                #[$stability]
                #[rustc_const_stable(feature = "const_nonzero_int_methods", since = "1.47.0")]
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
            impl From<$Ty> for $Int {
                doc_comment! {
                    concat!(
"Converts a `", stringify!($Ty), "` into an `", stringify!($Int), "`"),
                    #[inline]
                    fn from(nonzero: $Ty) -> Self {
                        nonzero.0
                    }
                }
            }

            #[stable(feature = "nonzero_bitor", since = "1.45.0")]
            impl BitOr for $Ty {
                type Output = Self;
                #[inline]
                fn bitor(self, rhs: Self) -> Self::Output {
                    // SAFETY: since `self` and `rhs` are both nonzero, the
                    // result of the bitwise-or will be nonzero.
                    unsafe { $Ty::new_unchecked(self.get() | rhs.get()) }
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
                    unsafe { $Ty::new_unchecked(self.get() | rhs) }
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
        )+
    }
}

nonzero_integers! {
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroU8(u8);
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroU16(u16);
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroU32(u32);
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroU64(u64);
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroU128(u128);
    #[stable(feature = "nonzero", since = "1.28.0")] NonZeroUsize(usize);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI8(i8);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI16(i16);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI32(i32);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI64(i64);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroI128(i128);
    #[stable(feature = "signed_nonzero", since = "1.34.0")] NonZeroIsize(isize);
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
                doc_comment! {
                    concat!("Returns the number of leading zeros in the binary representation of `self`.

On many architectures, this function can perform better than `leading_zeros()` on the underlying integer type, as special handling of zero can be avoided.

# Examples

Basic usage:

```
#![feature(nonzero_leading_trailing_zeros)]
let n = std::num::", stringify!($Ty), "::new(", stringify!($LeadingTestExpr), ").unwrap();

assert_eq!(n.leading_zeros(), 0);
```"),
                    #[unstable(feature = "nonzero_leading_trailing_zeros", issue = "79143")]
                    #[rustc_const_unstable(feature = "nonzero_leading_trailing_zeros", issue = "79143")]
                    #[inline]
                    pub const fn leading_zeros(self) -> u32 {
                        // SAFETY: since `self` can not be zero it is safe to call ctlz_nonzero
                        unsafe { intrinsics::ctlz_nonzero(self.0 as $Uint) as u32 }
                    }
                }

                doc_comment! {
                    concat!("Returns the number of trailing zeros in the binary representation
of `self`.

On many architectures, this function can perform better than `trailing_zeros()` on the underlying integer type, as special handling of zero can be avoided.

# Examples

Basic usage:

```
#![feature(nonzero_leading_trailing_zeros)]
let n = std::num::", stringify!($Ty), "::new(0b0101000).unwrap();

assert_eq!(n.trailing_zeros(), 3);
```"),
                    #[unstable(feature = "nonzero_leading_trailing_zeros", issue = "79143")]
                    #[rustc_const_unstable(feature = "nonzero_leading_trailing_zeros", issue = "79143")]
                    #[inline]
                    pub const fn trailing_zeros(self) -> u32 {
                        // SAFETY: since `self` can not be zero it is safe to call cttz_nonzero
                        unsafe { intrinsics::cttz_nonzero(self.0 as $Uint) as u32 }
                    }
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
            impl Div<$Ty> for $Int {
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
            impl Rem<$Ty> for $Int {
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
