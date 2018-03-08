// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Wrapping;

use ops::*;

#[allow(unused_macros)]
macro_rules! sh_impl_signed {
    ($t:ident, $f:ident) => (
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

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
        forward_ref_op_assign! { impl ShrAssign, shr_assign for Wrapping<$t>, $f }
    )
}

macro_rules! sh_impl_unsigned {
    ($t:ident, $f:ident) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline]
            fn shl(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_shl((other & self::shift_max::$t as $f) as u32))
            }
        }

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

        #[stable(feature = "op_assign_traits", since = "1.8.0")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
        forward_ref_op_assign! { impl ShrAssign, shr_assign for Wrapping<$t>, $f }
    )
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! sh_impl_all {
    ($($t:ident)*) => ($(
        //sh_impl_unsigned! { $t, u8 }
        //sh_impl_unsigned! { $t, u16 }
        //sh_impl_unsigned! { $t, u32 }
        //sh_impl_unsigned! { $t, u64 }
        sh_impl_unsigned! { $t, usize }

        //sh_impl_signed! { $t, i8 }
        //sh_impl_signed! { $t, i16 }
        //sh_impl_signed! { $t, i32 }
        //sh_impl_signed! { $t, i64 }
        //sh_impl_signed! { $t, isize }
    )*)
}

sh_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

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
            /// Returns the number of ones in the binary representation of
            /// `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i8> = Wrapping(-0b1000_0000);
            ///
            /// assert_eq!(n.count_ones(), 1);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            /// Returns the number of zeros in the binary representation of
            /// `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i8> = Wrapping(-0b1000_0000);
            ///
            /// assert_eq!(n.count_zeros(), 7);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }

            /// Returns the number of leading zeros in the binary representation
            /// of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i16> = Wrapping(-1);
            ///
            /// assert_eq!(n.leading_zeros(), 0);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            /// Returns the number of trailing zeros in the binary representation
            /// of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let n: Wrapping<i8> = Wrapping(-4);
            ///
            /// assert_eq!(n.trailing_zeros(), 2);
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn trailing_zeros(self) -> u32 {
                self.0.trailing_zeros()
            }

            /// Shifts the bits to the left by a specified amount, `n`,
            /// wrapping the truncated bits to the end of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as `>>`!
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
            pub fn rotate_left(self, n: u32) -> Self {
                Wrapping(self.0.rotate_left(n))
            }

            /// Shifts the bits to the right by a specified amount, `n`,
            /// wrapping the truncated bits to the beginning of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as `<<`!
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
            pub fn rotate_right(self, n: u32) -> Self {
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
            pub fn swap_bytes(self) -> Self {
                Wrapping(self.0.swap_bytes())
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
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            ///
            /// if cfg!(target_endian = "big") {
            ///     assert_eq!(Wrapping::<i64>::from_be(n), n);
            /// } else {
            ///     assert_eq!(Wrapping::<i64>::from_be(n), n.swap_bytes());
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn from_be(x: Self) -> Self {
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
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            ///
            /// if cfg!(target_endian = "little") {
            ///     assert_eq!(Wrapping::<i64>::from_le(n), n);
            /// } else {
            ///     assert_eq!(Wrapping::<i64>::from_le(n), n.swap_bytes());
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn from_le(x: Self) -> Self {
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
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            ///
            /// if cfg!(target_endian = "big") {
            ///     assert_eq!(n.to_be(), n);
            /// } else {
            ///     assert_eq!(n.to_be(), n.swap_bytes());
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn to_be(self) -> Self {
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
            /// let n: Wrapping<i64> = Wrapping(0x0123456789ABCDEF);
            ///
            /// if cfg!(target_endian = "little") {
            ///     assert_eq!(n.to_le(), n);
            /// } else {
            ///     assert_eq!(n.to_le(), n.swap_bytes());
            /// }
            /// ```
            #[inline]
            #[unstable(feature = "wrapping_int_impl", issue = "32463")]
            pub fn to_le(self) -> Self {
                Wrapping(self.0.to_le())
            }

            /// Raises self to the power of `exp`, using exponentiation by
            /// squaring.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// let x: Wrapping<i32> = Wrapping(2); // or any other integer type
            ///
            /// assert_eq!(x.pow(4), Wrapping(16));
            /// ```
            ///
            /// Results that are too large are wrapped:
            ///
            /// ```
            /// #![feature(wrapping_int_impl)]
            /// use std::num::Wrapping;
            ///
            /// // 5 ^ 4 = 625, which is too big for a u8
            /// let x: Wrapping<u8> = Wrapping(5);
            ///
            /// assert_eq!(x.pow(4).0, 113);
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
    pub use self::platform::isize;

    pub const u8: u32 = i8;
    pub const u16: u32 = i16;
    pub const u32: u32 = i32;
    pub const u64: u32 = i64;
    pub use self::platform::usize;
}
