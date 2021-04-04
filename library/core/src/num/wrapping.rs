//! Definitions of `Wrapping<T>`.

use crate::fmt;
use crate::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign};
use crate::ops::{BitXor, BitXorAssign, Div, DivAssign};
use crate::ops::{Mul, MulAssign, Neg, Not, Rem, RemAssign};
use crate::ops::{Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};

/// Provides intentionally-wrapped arithmetic on `T`.
///
/// Operations like `+` on `u32` values are intended to never overflow,
/// and in some debug configurations overflow is detected and results
/// in a panic. While most arithmetic falls into this category, some
/// code explicitly expects and relies upon modular arithmetic (e.g.,
/// hashing).
///
/// Wrapping arithmetic can be achieved either through methods like
/// `wrapping_add`, or through the `Wrapping<T>` type, which says that
/// all standard arithmetic operations on the underlying value are
/// intended to have wrapping semantics.
///
/// The underlying value can be retrieved through the `.0` index of the
/// `Wrapping` tuple.
///
/// # Examples
///
/// ```
/// use std::num::Wrapping;
///
/// let zero = Wrapping(0u32);
/// let one = Wrapping(1u32);
///
/// assert_eq!(u32::MAX, (zero - one).0);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
#[repr(transparent)]
pub struct Wrapping<T>(#[stable(feature = "rust1", since = "1.0.0")] pub T);

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_display", since = "1.10.0")]
impl<T: fmt::Display> fmt::Display for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::Binary> fmt::Binary for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::Octal> fmt::Octal for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::LowerHex> fmt::LowerHex for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[stable(feature = "wrapping_fmt", since = "1.11.0")]
impl<T: fmt::UpperHex> fmt::UpperHex for Wrapping<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[allow(unused_macros)]
macro_rules! sh_impl_signed {
    ($t:ident, $f:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn shl(self, other: $f) -> Wrapping<$t> {
                if other < 0 {
                    Wrapping(self.0.wrapping_shr((-other & self::shift_max::$t as $f) as u32))
                } else {
                    Wrapping(self.0.wrapping_shl((other & self::shift_max::$t as $f) as u32))
                }
            }
        }
        forward_ref_binop! { impl Shl, shl for Wrapping<$t>, $f,
        #[stable(feature = "wrapping_ref_ops", since = "1.39.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShlAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shl_assign(&mut self, other: $f) {
                *self = *self << other;
            }
        }
        forward_ref_op_assign! { impl ShlAssign, shl_assign for Wrapping<$t>, $f }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn shr(self, other: $f) -> Wrapping<$t> {
                if other < 0 {
                    Wrapping(self.0.wrapping_shl((-other & self::shift_max::$t as $f) as u32))
                } else {
                    Wrapping(self.0.wrapping_shr((other & self::shift_max::$t as $f) as u32))
                }
            }
        }
        forward_ref_binop! { impl Shr, shr for Wrapping<$t>, $f,
        #[stable(feature = "wrapping_ref_ops", since = "1.39.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
        forward_ref_op_assign! { impl ShrAssign, shr_assign for Wrapping<$t>, $f }
    };
}

macro_rules! sh_impl_unsigned {
    ($t:ident, $f:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn shl(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_shl((other & self::shift_max::$t as $f) as u32))
            }
        }
        forward_ref_binop! { impl Shl, shl for Wrapping<$t>, $f,
        #[stable(feature = "wrapping_ref_ops", since = "1.39.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShlAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shl_assign(&mut self, other: $f) {
                *self = *self << other;
            }
        }
        forward_ref_op_assign! { impl ShlAssign, shl_assign for Wrapping<$t>, $f }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn shr(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_shr((other & self::shift_max::$t as $f) as u32))
            }
        }
        forward_ref_binop! { impl Shr, shr for Wrapping<$t>, $f,
        #[stable(feature = "wrapping_ref_ops", since = "1.39.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
        forward_ref_op_assign! { impl ShrAssign, shr_assign for Wrapping<$t>, $f }
    };
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! sh_impl_all {
    ($($t:ident)*) => ($(
        //sh_impl_unsigned! { $t, u8 }
        //sh_impl_unsigned! { $t, u16 }
        //sh_impl_unsigned! { $t, u32 }
        //sh_impl_unsigned! { $t, u64 }
        //sh_impl_unsigned! { $t, u128 }
        sh_impl_unsigned! { $t, usize }

        //sh_impl_signed! { $t, i8 }
        //sh_impl_signed! { $t, i16 }
        //sh_impl_signed! { $t, i32 }
        //sh_impl_signed! { $t, i64 }
        //sh_impl_signed! { $t, i128 }
        //sh_impl_signed! { $t, isize }
    )*)
}

sh_impl_all! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }

// FIXME(30524): impl Op<T> for Wrapping<T>, impl OpAssign<T> for Wrapping<T>
macro_rules! wrapping_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Add for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn add(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_add(other.0))
            }
        }
        forward_ref_binop! { impl Add, add for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl AddAssign for Wrapping<$t> {
            #[inline]
            fn add_assign(&mut self, other: Wrapping<$t>) {
                *self = *self + other;
            }
        }
        forward_ref_op_assign! { impl AddAssign, add_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Sub for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn sub(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_sub(other.0))
            }
        }
        forward_ref_binop! { impl Sub, sub for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl SubAssign for Wrapping<$t> {
            #[inline]
            fn sub_assign(&mut self, other: Wrapping<$t>) {
                *self = *self - other;
            }
        }
        forward_ref_op_assign! { impl SubAssign, sub_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Mul for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn mul(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_mul(other.0))
            }
        }
        forward_ref_binop! { impl Mul, mul for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl MulAssign for Wrapping<$t> {
            #[inline]
            fn mul_assign(&mut self, other: Wrapping<$t>) {
                *self = *self * other;
            }
        }
        forward_ref_op_assign! { impl MulAssign, mul_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "wrapping_div", since = "1.3.0")]
        impl Div for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn div(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_div(other.0))
            }
        }
        forward_ref_binop! { impl Div, div for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl DivAssign for Wrapping<$t> {
            #[inline]
            fn div_assign(&mut self, other: Wrapping<$t>) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl DivAssign, div_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "wrapping_impls", since = "1.7.0")]
        impl Rem for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn rem(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_rem(other.0))
            }
        }
        forward_ref_binop! { impl Rem, rem for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl RemAssign for Wrapping<$t> {
            #[inline]
            fn rem_assign(&mut self, other: Wrapping<$t>) {
                *self = *self % other;
            }
        }
        forward_ref_op_assign! { impl RemAssign, rem_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Not for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn not(self) -> Wrapping<$t> {
                Wrapping(!self.0)
            }
        }
        forward_ref_unop! { impl Not, not for Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitXor for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn bitxor(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 ^ other.0)
            }
        }
        forward_ref_binop! { impl BitXor, bitxor for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitXorAssign for Wrapping<$t> {
            #[inline]
            fn bitxor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self ^ other;
            }
        }
        forward_ref_op_assign! { impl BitXorAssign, bitxor_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn bitor(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 | other.0)
            }
        }
        forward_ref_binop! { impl BitOr, bitor for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitOrAssign for Wrapping<$t> {
            #[inline]
            fn bitor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self | other;
            }
        }
        forward_ref_op_assign! { impl BitOrAssign, bitor_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitAnd for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn bitand(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 & other.0)
            }
        }
        forward_ref_binop! { impl BitAnd, bitand for Wrapping<$t>, Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl BitAndAssign for Wrapping<$t> {
            #[inline]
            fn bitand_assign(&mut self, other: Wrapping<$t>) {
                *self = *self & other;
            }
        }
        forward_ref_op_assign! { impl BitAndAssign, bitand_assign for Wrapping<$t>, Wrapping<$t> }

        #[stable(feature = "wrapping_neg", since = "1.10.0")]
        impl Neg for Wrapping<$t> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Wrapping(0) - self
            }
        }
        forward_ref_unop! { impl Neg, neg for Wrapping<$t>,
                #[stable(feature = "wrapping_ref", since = "1.14.0")] }

    )*)
}

wrapping_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

macro_rules! wrapping_int_impl {
    ($($t:ty)*) => ($(
        impl Wrapping<$t> {
            /// Returns the smallest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(<Wrapping<", stringify!($t), ">>::MIN, Wrapping(", stringify!($t), "::MIN));")]
            /// ```
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const MIN: Self = Self(<$t>::MIN);

            /// Returns the largest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(<Wrapping<", stringify!($t), ">>::MAX, Wrapping(", stringify!($t), "::MAX));")]
            /// ```
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const MAX: Self = Self(<$t>::MAX);

            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0b01001100", stringify!($t), ");")]
            ///
            /// assert_eq!(n.count_ones(), 3);
            /// ```
            #[inline]
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(Wrapping(!0", stringify!($t), ").count_zeros(), 0);")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0b0101000", stringify!($t), ");")]
            ///
            /// assert_eq!(n.trailing_zeros(), 3);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn trailing_zeros(self) -> u32 {
                self.0.trailing_zeros()
            }

            /// Shifts the bits to the left by a specified amount, `n`,
            /// wrapping the truncated bits to the end of the resulting
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            /// let m: Wrapping<i64> = Wrapping(-0x76543210FEDCBA99);
            ///
            /// assert_eq!(n.rotate_left(32), m);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn rotate_left(self, n: u32) -> Self {
                Wrapping(self.0.rotate_left(n))
            }

            /// Shifts the bits to the right by a specified amount, `n`,
            /// wrapping the truncated bits to the beginning of the resulting
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            /// let m: Wrapping<i64> = Wrapping(-0xFEDCBA987654322);
            ///
            /// assert_eq!(n.rotate_right(4), m);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn rotate_right(self, n: u32) -> Self {
                Wrapping(self.0.rotate_right(n))
            }

            /// Reverses the byte order of the integer.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i16> = Wrapping(0b0000000_01010101);
            /// assert_eq!(n, Wrapping(85));
            ///
            /// let m = n.swap_bytes();
            ///
            /// assert_eq!(m, Wrapping(0b01010101_00000000));
            /// assert_eq!(m, Wrapping(21760));
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn swap_bytes(self) -> Self {
                Wrapping(self.0.swap_bytes())
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
            /// use std::num::Wrapping;
            ///
            /// let n = Wrapping(0b0000000_01010101i16);
            /// assert_eq!(n, Wrapping(85));
            ///
            /// let m = n.reverse_bits();
            ///
            /// assert_eq!(m.0 as u16, 0b10101010_00000000);
            /// assert_eq!(m, Wrapping(-22016));
            /// ```
            #[stable(feature = "reverse_bits", since = "1.37.0")]
            #[rustc_const_stable(feature = "const_reverse_bits", since = "1.37.0")]
            #[inline]
            #[must_use]
            pub const fn reverse_bits(self) -> Self {
                Wrapping(self.0.reverse_bits())
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "big") {
            #[doc = concat!("    assert_eq!(<Wrapping<", stringify!($t), ">>::from_be(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(<Wrapping<", stringify!($t), ">>::from_be(n), n.swap_bytes())")]
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn from_be(x: Self) -> Self {
                Wrapping(<$t>::from_be(x.0))
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "little") {
            #[doc = concat!("    assert_eq!(<Wrapping<", stringify!($t), ">>::from_le(n), n)")]
            /// } else {
            #[doc = concat!("    assert_eq!(<Wrapping<", stringify!($t), ">>::from_le(n), n.swap_bytes())")]
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn from_le(x: Self) -> Self {
                Wrapping(<$t>::from_le(x.0))
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "big") {
            ///     assert_eq!(n.to_be(), n)
            /// } else {
            ///     assert_eq!(n.to_be(), n.swap_bytes())
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn to_be(self) -> Self {
                Wrapping(self.0.to_be())
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(0x1A", stringify!($t), ");")]
            ///
            /// if cfg!(target_endian = "little") {
            ///     assert_eq!(n.to_le(), n)
            /// } else {
            ///     assert_eq!(n.to_le(), n.swap_bytes())
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn to_le(self) -> Self {
                Wrapping(self.0.to_le())
            }

            /// Raises self to the power of `exp`, using exponentiation by squaring.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(Wrapping(3", stringify!($t), ").pow(4), Wrapping(81));")]
            /// ```
            ///
            /// Results that are too large are wrapped:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// assert_eq!(Wrapping(3i8).pow(5), Wrapping(-13));
            /// assert_eq!(Wrapping(3i8).pow(6), Wrapping(-39));
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn pow(self, exp: u32) -> Self {
                Wrapping(self.0.wrapping_pow(exp))
            }
        }
    )*)
}

wrapping_int_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

macro_rules! wrapping_int_impl_signed {
    ($($t:ty)*) => ($(
        impl Wrapping<$t> {
            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(", stringify!($t), "::MAX) >> 2;")]
            ///
            /// assert_eq!(n.leading_zeros(), 3);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            /// Computes the absolute value of `self`, wrapping around at
            /// the boundary of the type.
            ///
            /// The only case where such wrapping can occur is when one takes the absolute value of the negative
            /// minimal value for the type this is a positive value that is too large to represent in the type. In
            /// such a case, this function returns `MIN` itself.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(Wrapping(100", stringify!($t), ").abs(), Wrapping(100));")]
            #[doc = concat!("assert_eq!(Wrapping(-100", stringify!($t), ").abs(), Wrapping(100));")]
            #[doc = concat!("assert_eq!(Wrapping(", stringify!($t), "::MIN).abs(), Wrapping(", stringify!($t), "::MIN));")]
            /// assert_eq!(Wrapping(-128i8).abs().0 as u8, 128u8);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn abs(self) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_abs())
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(Wrapping(10", stringify!($t), ").signum(), Wrapping(1));")]
            #[doc = concat!("assert_eq!(Wrapping(0", stringify!($t), ").signum(), Wrapping(0));")]
            #[doc = concat!("assert_eq!(Wrapping(-10", stringify!($t), ").signum(), Wrapping(-1));")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn signum(self) -> Wrapping<$t> {
                Wrapping(self.0.signum())
            }

            /// Returns `true` if `self` is positive and `false` if the number is zero or
            /// negative.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert!(Wrapping(10", stringify!($t), ").is_positive());")]
            #[doc = concat!("assert!(!Wrapping(-10", stringify!($t), ").is_positive());")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert!(Wrapping(-10", stringify!($t), ").is_negative());")]
            #[doc = concat!("assert!(!Wrapping(10", stringify!($t), ").is_negative());")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub const fn is_negative(self) -> bool {
                self.0.is_negative()
            }
        }
    )*)
}

wrapping_int_impl_signed! { isize i8 i16 i32 i64 i128 }

macro_rules! wrapping_int_impl_unsigned {
    ($($t:ty)*) => ($(
        impl Wrapping<$t> {
            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("let n = Wrapping(", stringify!($t), "::MAX) >> 2;")]
            ///
            /// assert_eq!(n.leading_zeros(), 2);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
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
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert!(Wrapping(16", stringify!($t), ").is_power_of_two());")]
            #[doc = concat!("assert!(!Wrapping(10", stringify!($t), ").is_power_of_two());")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn is_power_of_two(self) -> bool {
                self.0.is_power_of_two()
            }

            /// Returns the smallest power of two greater than or equal to `self`.
            ///
            /// When return value overflows (i.e., `self > (1 << (N-1))` for type
            /// `uN`), overflows to `2^N = 0`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_next_power_of_two)]
            /// use std::num::Wrapping;
            ///
            #[doc = concat!("assert_eq!(Wrapping(2", stringify!($t), ").next_power_of_two(), Wrapping(2));")]
            #[doc = concat!("assert_eq!(Wrapping(3", stringify!($t), ").next_power_of_two(), Wrapping(4));")]
            #[doc = concat!("assert_eq!(Wrapping(200_u8).next_power_of_two(), Wrapping(0));")]
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_next_power_of_two", issue = "32463",
                       reason = "needs decision on wrapping behaviour")]
            pub fn next_power_of_two(self) -> Self {
                Wrapping(self.0.wrapping_next_power_of_two())
            }
        }
    )*)
}

wrapping_int_impl_unsigned! { usize u8 u16 u32 u64 u128 }

mod shift_max {
    #![allow(non_upper_case_globals)]

    #[cfg(target_pointer_width = "16")]
    mod platform {
        pub const usize: u32 = super::u16;
        pub const isize: u32 = super::i16;
    }

    #[cfg(target_pointer_width = "32")]
    mod platform {
        pub const usize: u32 = super::u32;
        pub const isize: u32 = super::i32;
    }

    #[cfg(target_pointer_width = "64")]
    mod platform {
        pub const usize: u32 = super::u64;
        pub const isize: u32 = super::i64;
    }

    pub const i8: u32 = (1 << 3) - 1;
    pub const i16: u32 = (1 << 4) - 1;
    pub const i32: u32 = (1 << 5) - 1;
    pub const i64: u32 = (1 << 6) - 1;
    pub const i128: u32 = (1 << 7) - 1;
    pub use self::platform::isize;

    pub const u8: u32 = i8;
    pub const u16: u32 = i16;
    pub const u32: u32 = i32;
    pub const u64: u32 = i64;
    pub const u128: u32 = i128;
    pub use self::platform::usize;
}
