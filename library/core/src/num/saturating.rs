//! Definitions of `Saturating<T>`.

use crate::fmt;
use crate::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign};
use crate::ops::{BitXor, BitXorAssign, Div, DivAssign};
use crate::ops::{Mul, MulAssign, Neg, Not};
use crate::ops::{Sub, SubAssign};

/// Provides intentionally-wrapped arithmetic on `T`.
///
/// Operations like `+` on `u32` values are intended to never overflow,
/// and in some debug configurations overflow is detected and results
/// in a panic. While most arithmetic falls into this category, some
/// code explicitly expects and relies upon modular arithmetic (e.g.,
/// hashing).
///
/// Saturating arithmetic can be achieved either through methods like
/// `saturating_add`, or through the `Saturating<T>` type, which says that
/// all standard arithmetic operations on the underlying value are
/// intended to have saturating semantics.
///
/// The underlying value can be retrieved through the `.0` index of the
/// `Saturating` tuple.
///
/// # Examples
///
/// ```
/// #![feature(saturating_int_impl)]
/// use std::num::Saturating;
///
/// let max = Saturating(u32::MAX);
/// let one = Saturating(1u32);
///
/// assert_eq!(u32::MAX, (max + one).0);
/// ```
#[unstable(feature = "saturating_int_impl", issue = "87920")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
#[repr(transparent)]
pub struct Saturating<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "saturating_display", since = "1.10.0")]
impl<T: fmt::Display> fmt::Display for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "saturating_fmt", since = "1.11.0")]
impl<T: fmt::Binary> fmt::Binary for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "saturating_fmt", since = "1.11.0")]
impl<T: fmt::Octal> fmt::Octal for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "saturating_fmt", since = "1.11.0")]
impl<T: fmt::LowerHex> fmt::LowerHex for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "saturating_fmt", since = "1.11.0")]
impl<T: fmt::UpperHex> fmt::UpperHex for Saturating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// FIXME(30524): impl Op<T> for Saturating<T>, impl OpAssign<T> for Saturating<T>
macro_rules! saturating_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Add for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn add(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0.saturating_add(other.0))
            }
        }
        forward_ref_binop! { impl Add, add for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl AddAssign for Saturating<$t> {
            #[inline]
            fn add_assign(&mut self, other: Saturating<$t>) {
                *self = *self + other;
            }
        }
        forward_ref_op_assign! { impl AddAssign, add_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Sub for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn sub(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0.saturating_sub(other.0))
            }
        }
        forward_ref_binop! { impl Sub, sub for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl SubAssign for Saturating<$t> {
            #[inline]
            fn sub_assign(&mut self, other: Saturating<$t>) {
                *self = *self - other;
            }
        }
        forward_ref_op_assign! { impl SubAssign, sub_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Mul for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn mul(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0.saturating_mul(other.0))
            }
        }
        forward_ref_binop! { impl Mul, mul for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl MulAssign for Saturating<$t> {
            #[inline]
            fn mul_assign(&mut self, other: Saturating<$t>) {
                *self = *self * other;
            }
        }
        forward_ref_op_assign! { impl MulAssign, mul_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "saturating_div", since = "1.3.0")]
        impl Div for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn div(self, other: Saturating<$t>) -> Saturating<$t> {
                // saturating div is the default behavior?
                Saturating(self.0.div(other.0))
            }
        }
        forward_ref_binop! { impl Div, div for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl DivAssign for Saturating<$t> {
            #[inline]
            fn div_assign(&mut self, other: Saturating<$t>) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl DivAssign, div_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Not for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn not(self) -> Saturating<$t> {
                Saturating(!self.0)
            }
        }
        forward_ref_unop! { impl Not, not for Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitXor for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn bitxor(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0 ^ other.0)
            }
        }
        forward_ref_binop! { impl BitXor, bitxor for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitXorAssign for Saturating<$t> {
            #[inline]
            fn bitxor_assign(&mut self, other: Saturating<$t>) {
                *self = *self ^ other;
            }
        }
        forward_ref_op_assign! { impl BitXorAssign, bitxor_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn bitor(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0 | other.0)
            }
        }
        forward_ref_binop! { impl BitOr, bitor for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitOrAssign for Saturating<$t> {
            #[inline]
            fn bitor_assign(&mut self, other: Saturating<$t>) {
                *self = *self | other;
            }
        }
        forward_ref_op_assign! { impl BitOrAssign, bitor_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitAnd for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline]
            fn bitand(self, other: Saturating<$t>) -> Saturating<$t> {
                Saturating(self.0 & other.0)
            }
        }
        forward_ref_binop! { impl BitAnd, bitand for Saturating<$t>, Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitAndAssign for Saturating<$t> {
            #[inline]
            fn bitand_assign(&mut self, other: Saturating<$t>) {
                *self = *self & other;
            }
        }
        forward_ref_op_assign! { impl BitAndAssign, bitand_assign for Saturating<$t>, Saturating<$t> }

        #[stable(feature = "saturating_neg", since = "1.45.0")]
        impl Neg for Saturating<$t> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Saturating(0) - self
            }
        }
        forward_ref_unop! { impl Neg, neg for Saturating<$t>,
                #[stable(feature = "saturating_ref", since = "1.14.0")] }

    )*)
}

saturating_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

macro_rules! saturating_int_impl {
    ($($t:ty)*) => ($(
        impl Saturating<$t> {
            /// Returns the smallest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(<Saturating<", stringify!($t), ">>::MIN, Saturating(", stringify!($t), "::MIN));")]
            /// ```
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const MIN: Self = Self(<$t>::MIN);

            /// Returns the largest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(<Saturating<", stringify!($t), ">>::MAX, Saturating(", stringify!($t), "::MAX));")]
            /// ```
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const MAX: Self = Self(<$t>::MAX);

            /// Returns the size of this integer type in bits.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(<Saturating<", stringify!($t), ">>::BITS, ", stringify!($t), "::BITS);")]
            /// ```
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const BITS: u32 = <$t>::BITS;

            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0b01001100", stringify!($t), ");")]
            ///
            /// assert_eq!(n.count_ones(), 3);
            /// ```
            #[inline]
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            /// Returns the number of zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(Saturating(!0", stringify!($t), ").count_zeros(), 0);")]
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }

            /// Returns the number of trailing zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0b0101000", stringify!($t), ");")]
            ///
            /// assert_eq!(n.trailing_zeros(), 3);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn trailing_zeros(self) -> u32 {
                self.0.trailing_zeros()
            }

            /// Shifts the bits to the left by a specified amount, `n`,
            /// saturating the truncated bits to the end of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as the `<<` shifting
            /// operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            /// let n: Saturating<i64> = Saturating(0x0123456789ABCDEF);
            /// let m: Saturating<i64> = Saturating(-0x76543210FEDCBA99);
            ///
            /// assert_eq!(n.rotate_left(32), m);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn rotate_left(self, n: u32) -> Self {
                Saturating(self.0.rotate_left(n))
            }

            /// Shifts the bits to the right by a specified amount, `n`,
            /// saturating the truncated bits to the beginning of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as the `>>` shifting
            /// operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            /// let n: Saturating<i64> = Saturating(0x0123456789ABCDEF);
            /// let m: Saturating<i64> = Saturating(-0xFEDCBA987654322);
            ///
            /// assert_eq!(n.rotate_right(4), m);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn rotate_right(self, n: u32) -> Self {
                Saturating(self.0.rotate_right(n))
            }

            /// Reverses the byte order of the integer.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            /// let n: Saturating<i16> = Saturating(0b0000000_01010101);
            /// assert_eq!(n, Saturating(85));
            ///
            /// let m = n.swap_bytes();
            ///
            /// assert_eq!(m, Saturating(0b01010101_00000000));
            /// assert_eq!(m, Saturating(21760));
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn swap_bytes(self) -> Self {
                Saturating(self.0.swap_bytes())
            }

            /// Reverses the bit pattern of the integer.
            ///
            /// # Examples
            ///
            /// Please note that this example is shared between integer types.
            /// Which explains why `i16` is used here.
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            /// let n = Saturating(0b0000000_01010101i16);
            /// assert_eq!(n, Saturating(85));
            ///
            /// let m = n.reverse_bits();
            ///
            /// assert_eq!(m.0 as u16, 0b10101010_00000000);
            /// assert_eq!(m, Saturating(-22016));
            /// ```
            #[stable(feature = "reverse_bits", since = "1.37.0")]
            #[rustc_const_stable(feature = "const_reverse_bits", since = "1.37.0")]
            #[inline]
            #[must_use]
            pub const fn reverse_bits(self) -> Self {
                Saturating(self.0.reverse_bits())
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
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "big") {
            #[doc = concat!("    assert_eq!(<Saturating<", stringify!($t), ">>::from_be(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(<Saturating<", stringify!($t), ">>::from_be(n), n.swap_bytes())")]
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn from_be(x: Self) -> Self {
                Saturating(<$t>::from_be(x.0))
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
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "little") {
            #[doc = concat!("    assert_eq!(<Saturating<", stringify!($t), ">>::from_le(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(<Saturating<", stringify!($t), ">>::from_le(n), n.swap_bytes())")]
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn from_le(x: Self) -> Self {
                Saturating(<$t>::from_le(x.0))
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
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "big") {
            ///     assert_eq!(n.to_be(), n)
            /// } else {
            ///     assert_eq!(n.to_be(), n.swap_bytes())
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn to_be(self) -> Self {
                Saturating(self.0.to_be())
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
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "little") {
            ///     assert_eq!(n.to_le(), n)
            /// } else {
            ///     assert_eq!(n.to_le(), n.swap_bytes())
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn to_le(self) -> Self {
                Saturating(self.0.to_le())
            }

            /// Raises self to the power of `exp`, using exponentiation by squaring.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(Saturating(3", stringify!($t), ").pow(4), Saturating(81));")]
            /// ```
            ///
            /// Results that are too large are wrapped:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            /// assert_eq!(Saturating(3i8).pow(5), Saturating(127));
            /// assert_eq!(Saturating(3i8).pow(6), Saturating(127));
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub fn pow(self, exp: u32) -> Self {
                Saturating(self.0.saturating_pow(exp))
            }
        }
    )*)
}

saturating_int_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

macro_rules! saturating_int_impl_signed {
    ($($t:ty)*) => ($(
        impl Saturating<$t> {
            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(", stringify!($t), "::MAX) / Saturating(4", stringify!($t), ");")]
            ///
            /// assert_eq!(n.leading_zeros(), 3);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            /// Saturating absolute value. Computes `self.abs()`, returning `MAX` if `self == MIN`
            /// instead of overflowing.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(Saturating(100", stringify!($t), ").abs(), Saturating(100));")]
            #[doc = concat!("assert_eq!(Saturating(-100", stringify!($t), ").abs(), Saturating(100));")]
            #[doc = concat!("assert_eq!(Saturating(", stringify!($t), "::MIN).abs(), Saturating((", stringify!($t), "::MIN + 1).abs()));")]
            #[doc = concat!("assert_eq!(Saturating(", stringify!($t), "::MIN).abs(), Saturating(", stringify!($t), "::MIN.saturating_abs()));")]
            /// assert_eq!(Saturating(-128i8).abs().0 as u8, i8::MAX as u8);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub fn abs(self) -> Saturating<$t> {
                Saturating(self.0.saturating_abs())
            }

            /// Returns a number representing sign of `self`.
            ///
            ///  - `0` if the number is zero
            ///  - `1` if the number is positive
            ///  - `-1` if the number is negative
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert_eq!(Saturating(10", stringify!($t), ").signum(), Saturating(1));")]
            #[doc = concat!("assert_eq!(Saturating(0", stringify!($t), ").signum(), Saturating(0));")]
            #[doc = concat!("assert_eq!(Saturating(-10", stringify!($t), ").signum(), Saturating(-1));")]
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub fn signum(self) -> Saturating<$t> {
                Saturating(self.0.signum())
            }

            /// Returns `true` if `self` is positive and `false` if the number is zero or
            /// negative.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert!(Saturating(10", stringify!($t), ").is_positive());")]
            #[doc = concat!("assert!(!Saturating(-10", stringify!($t), ").is_positive());")]
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn is_positive(self) -> bool {
                self.0.is_positive()
            }

            /// Returns `true` if `self` is negative and `false` if the number is zero or
            /// positive.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert!(Saturating(-10", stringify!($t), ").is_negative());")]
            #[doc = concat!("assert!(!Saturating(10", stringify!($t), ").is_negative());")]
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn is_negative(self) -> bool {
                self.0.is_negative()
            }
        }
    )*)
}

saturating_int_impl_signed! { isize i8 i16 i32 i64 i128 }

macro_rules! saturating_int_impl_unsigned {
    ($($t:ty)*) => ($(
        impl Saturating<$t> {
            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("let n = Saturating(", stringify!($t), "::MAX) / Saturating(4", stringify!($t), ");")]
            ///
            /// assert_eq!(n.leading_zeros(), 2);
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub const fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            /// Returns `true` if and only if `self == 2^k` for some `k`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(saturating_int_impl)]
            /// use std::num::Saturating;
            ///
            #[doc = concat!("assert!(Saturating(16", stringify!($t), ").is_power_of_two());")]
            #[doc = concat!("assert!(!Saturating(10", stringify!($t), ").is_power_of_two());")]
            /// ```
            #[inline]
            #[unstable(feature = "saturating_int_impl", issue = "87920")]
            pub fn is_power_of_two(self) -> bool {
                self.0.is_power_of_two()
            }

        }
    )*)
}

saturating_int_impl_unsigned! { usize u8 u16 u32 u64 u128 }
