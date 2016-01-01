// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]
#![unstable(feature = "wrapping", reason = "may be removed or relocated",
            issue = "27755")]

use intrinsics::{add_with_overflow, sub_with_overflow, mul_with_overflow};

use super::Wrapping;

use ops::*;

use ::{i8, i16, i32, i64, isize};

pub trait OverflowingOps {
    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);

    fn overflowing_div(self, rhs: Self) -> (Self, bool);
    fn overflowing_rem(self, rhs: Self) -> (Self, bool);
    fn overflowing_neg(self) -> (Self, bool);

    fn overflowing_shl(self, rhs: u32) -> (Self, bool);
    fn overflowing_shr(self, rhs: u32) -> (Self, bool);
}

macro_rules! sh_impl_signed {
    ($t:ident, $f:ident) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shl(self, other: $f) -> Wrapping<$t> {
                if other < 0 {
                    Wrapping(self.0 >> (-other & self::shift_max::$t as $f))
                } else {
                    Wrapping(self.0 << (other & self::shift_max::$t as $f))
                }
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl ShlAssign<$f> for Wrapping<$t> {
            #[inline(always)]
            fn shl_assign(&mut self, other: $f) {
                *self = *self << other;
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shr(self, other: $f) -> Wrapping<$t> {
                if other < 0 {
                    Wrapping(self.0 << (-other & self::shift_max::$t as $f))
                } else {
                    Wrapping(self.0 >> (other & self::shift_max::$t as $f))
                }
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline(always)]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
    )
}

macro_rules! sh_impl_unsigned {
    ($t:ident, $f:ident) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shl(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0 << (other & self::shift_max::$t as $f))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl ShlAssign<$f> for Wrapping<$t> {
            #[inline(always)]
            fn shl_assign(&mut self, other: $f) {
                *self = *self << other;
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shr(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0 >> (other & self::shift_max::$t as $f))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl ShrAssign<$f> for Wrapping<$t> {
            #[inline(always)]
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
    )
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! sh_impl_all {
    ($($t:ident)*) => ($(
        sh_impl_unsigned! { $t, u8 }
        sh_impl_unsigned! { $t, u16 }
        sh_impl_unsigned! { $t, u32 }
        sh_impl_unsigned! { $t, u64 }
        sh_impl_unsigned! { $t, usize }

        sh_impl_signed! { $t, i8 }
        sh_impl_signed! { $t, i16 }
        sh_impl_signed! { $t, i32 }
        sh_impl_signed! { $t, i64 }
        sh_impl_signed! { $t, isize }
    )*)
}

sh_impl_all! { u8 u16 u32 u64 usize i8 i16 i32 i64 isize }

macro_rules! wrapping_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Add for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn add(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_add(other.0))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Add<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn add(self, other: $t) -> Wrapping<$t> {
                self + Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl AddAssign for Wrapping<$t> {
            #[inline(always)]
            fn add_assign(&mut self, other: Wrapping<$t>) {
                *self = *self + other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl AddAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn add_assign(&mut self, other: $t) {
                self.add_assign(Wrapping(other))
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Sub for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn sub(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_sub(other.0))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Sub<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn sub(self, other: $t) -> Wrapping<$t> {
                self - Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl SubAssign for Wrapping<$t> {
            #[inline(always)]
            fn sub_assign(&mut self, other: Wrapping<$t>) {
                *self = *self - other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl SubAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn sub_assign(&mut self, other: $t) {
                self.sub_assign(Wrapping(other))
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Mul for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn mul(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_mul(other.0))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Mul<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn mul(self, other: $t) -> Wrapping<$t> {
                self * Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl MulAssign for Wrapping<$t> {
            #[inline(always)]
            fn mul_assign(&mut self, other: Wrapping<$t>) {
                *self = *self * other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl MulAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn mul_assign(&mut self, other: $t) {
                self.mul_assign(Wrapping(other))
            }
        }

        #[stable(feature = "wrapping_div", since = "1.3.0")]
        impl Div for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn div(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_div(other.0))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Div<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn div(self, other: $t) -> Wrapping<$t> {
                self / Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl DivAssign for Wrapping<$t> {
            #[inline(always)]
            fn div_assign(&mut self, other: Wrapping<$t>) {
                *self = *self / other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl DivAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn div_assign(&mut self, other: $t) {
                self.div_assign(Wrapping(other))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Rem for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn rem(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_rem(other.0))
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl Rem<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn rem(self, other: $t) -> Wrapping<$t> {
                self % Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl RemAssign for Wrapping<$t> {
            #[inline(always)]
            fn rem_assign(&mut self, other: Wrapping<$t>) {
                *self = *self % other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl RemAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn rem_assign(&mut self, other: $t) {
                self.rem_assign(Wrapping(other))
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Not for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn not(self) -> Wrapping<$t> {
                Wrapping(!self.0)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitXor for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitxor(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 ^ other.0)
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitXor<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitxor(self, other: $t) -> Wrapping<$t> {
                self ^ Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitXorAssign for Wrapping<$t> {
            #[inline(always)]
            fn bitxor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self ^ other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitXorAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn bitxor_assign(&mut self, other: $t) {
                self.bitxor_assign(Wrapping(other))
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitor(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 | other.0)
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitOr<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitor(self, other: $t) -> Wrapping<$t> {
                self | Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitOrAssign for Wrapping<$t> {
            #[inline(always)]
            fn bitor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self | other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitOrAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn bitor_assign(&mut self, other: $t) {
                self.bitor_assign(Wrapping(other))
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitAnd for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitand(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 & other.0)
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitAnd<$t> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitand(self, other: $t) -> Wrapping<$t> {
                self & Wrapping(other)
            }
        }

        #[unstable(feature = "op_assign_traits", reason = "recently added", issue = "28235")]
        impl BitAndAssign for Wrapping<$t> {
            #[inline(always)]
            fn bitand_assign(&mut self, other: Wrapping<$t>) {
                *self = *self & other;
            }
        }

        #[unstable(feature = "wrapping_impls", reason = "recently added", issue = "30524")]
        impl BitAndAssign<$t> for Wrapping<$t> {
            #[inline(always)]
            fn bitand_assign(&mut self, other: $t) {
                self.bitand_assign(Wrapping(other))
            }
        }
    )*)
}

wrapping_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

mod shift_max {
    #![allow(non_upper_case_globals)]

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

    pub const  i8: u32 = (1 << 3) - 1;
    pub const i16: u32 = (1 << 4) - 1;
    pub const i32: u32 = (1 << 5) - 1;
    pub const i64: u32 = (1 << 6) - 1;
    pub use self::platform::isize;

    pub const  u8: u32 = i8;
    pub const u16: u32 = i16;
    pub const u32: u32 = i32;
    pub const u64: u32 = i64;
    pub use self::platform::usize;
}

macro_rules! signed_overflowing_impl {
    ($($t:ident)*) => ($(
        impl OverflowingOps for $t {
            #[inline(always)]
            fn overflowing_add(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    add_with_overflow(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_sub(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    sub_with_overflow(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_mul(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    mul_with_overflow(self, rhs)
                }
            }

            #[inline(always)]
            fn overflowing_div(self, rhs: $t) -> ($t, bool) {
                if self == $t::MIN && rhs == -1 {
                    (self, true)
                } else {
                    (self/rhs, false)
                }
            }
            #[inline(always)]
            fn overflowing_rem(self, rhs: $t) -> ($t, bool) {
                if self == $t::MIN && rhs == -1 {
                    (0, true)
                } else {
                    (self % rhs, false)
                }
            }

            #[inline(always)]
            fn overflowing_shl(self, rhs: u32) -> ($t, bool) {
                (self << (rhs & self::shift_max::$t),
                 (rhs > self::shift_max::$t))
            }
            #[inline(always)]
            fn overflowing_shr(self, rhs: u32) -> ($t, bool) {
                (self >> (rhs & self::shift_max::$t),
                 (rhs > self::shift_max::$t))
            }

            #[inline(always)]
            fn overflowing_neg(self) -> ($t, bool) {
                if self == $t::MIN {
                    ($t::MIN, true)
                } else {
                    (-self, false)
                }
            }
        }
    )*)
}

macro_rules! unsigned_overflowing_impl {
    ($($t:ident)*) => ($(
        impl OverflowingOps for $t {
            #[inline(always)]
            fn overflowing_add(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    add_with_overflow(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_sub(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    sub_with_overflow(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_mul(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    mul_with_overflow(self, rhs)
                }
            }

            #[inline(always)]
            fn overflowing_div(self, rhs: $t) -> ($t, bool) {
                (self/rhs, false)
            }
            #[inline(always)]
            fn overflowing_rem(self, rhs: $t) -> ($t, bool) {
                (self % rhs, false)
            }

            #[inline(always)]
            fn overflowing_shl(self, rhs: u32) -> ($t, bool) {
                (self << (rhs & self::shift_max::$t),
                 (rhs > self::shift_max::$t))
            }
            #[inline(always)]
            fn overflowing_shr(self, rhs: u32) -> ($t, bool) {
                (self >> (rhs & self::shift_max::$t),
                 (rhs > self::shift_max::$t))
            }

            #[inline(always)]
            fn overflowing_neg(self) -> ($t, bool) {
                ((!self).wrapping_add(1), true)
            }
        }
    )*)
}

signed_overflowing_impl! { i8 i16 i32 i64 isize }
unsigned_overflowing_impl! { u8 u16 u32 u64 usize }
