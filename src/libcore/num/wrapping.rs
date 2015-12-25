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

pub use intrinsics::{add_with_overflow, sub_with_overflow, mul_with_overflow};

use super::Wrapping;

use ops::*;

use ::{i8,i16,i32,i64};

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

macro_rules! sh_impl {
    ($t:ty, $f:ty) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shl<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shl(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0 << other)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Shr<$f> for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn shr(self, other: $f) -> Wrapping<$t> {
                Wrapping(self.0 >> other)
            }
        }
    )
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! sh_impl_all {
    ($($t:ty)*) => ($(
        // sh_impl! { $t, u8 }
        // sh_impl! { $t, u16 }
        // sh_impl! { $t, u32 }
        // sh_impl! { $t, u64 }
        sh_impl! { $t, usize }

        // sh_impl! { $t, i8 }
        // sh_impl! { $t, i16 }
        // sh_impl! { $t, i32 }
        // sh_impl! { $t, i64 }
        // sh_impl! { $t, isize }
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

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Sub for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn sub(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_sub(other.0))
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

        #[stable(feature = "wrapping_div", since = "1.3.0")]
        impl Div for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn div(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0.wrapping_div(other.0))
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

        #[stable(feature = "rust1", since = "1.0.0")]
        impl BitOr for Wrapping<$t> {
            type Output = Wrapping<$t>;

            #[inline(always)]
            fn bitor(self, other: Wrapping<$t>) -> Wrapping<$t> {
                Wrapping(self.0 | other.0)
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

        impl AddAssign for Wrapping<$t> {
            fn add_assign(&mut self, other: Wrapping<$t>) {
                *self = *self + other;
            }
        }

        impl SubAssign for Wrapping<$t> {
            fn sub_assign(&mut self, other: Wrapping<$t>) {
                *self = *self - other;
            }
        }

        impl MulAssign for Wrapping<$t> {
            fn mul_assign(&mut self, other: Wrapping<$t>) {
                *self = *self * other;
            }
        }

        impl DivAssign for Wrapping<$t> {
            fn div_assign(&mut self, other: Wrapping<$t>) {
                *self = *self / other;
            }
        }

        /*
        impl RemAssign for Wrapping<$t> {
            fn rem_assign(&mut self, other: Wrapping<$t>) {
                *self = *self % other;
            }
        }
        */

        impl BitAndAssign for Wrapping<$t> {
            fn bitand_assign(&mut self, other: Wrapping<$t>) {
                *self = *self & other;
            }
        }

        impl BitOrAssign for Wrapping<$t> {
            fn bitor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self | other;
            }
        }

        impl BitXorAssign for Wrapping<$t> {
            fn bitxor_assign(&mut self, other: Wrapping<$t>) {
                *self = *self ^ other;
            }
        }
    )*)
}

wrapping_impl! { usize u8 u16 u32 u64 isize i8 i16 i32 i64 }


macro_rules! wrapping_impl_all {
    ($($t:ty)*) => ($(
        impl ShlAssign<$f> for Wrapping<$t> {
            fn shl_assign(&mut self, other: $f) {
                *self = *self << other;
            }
        }

        impl ShrAssign<$f> for Wrapping<$t> {
            fn shr_assign(&mut self, other: $f) {
                *self = *self >> other;
            }
        }
    )*)
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! sh_impl_all {
    ($($t:ty)*) => ($(
        // sh_impl! { $t, u8 }
        // sh_impl! { $t, u16 }
        // sh_impl! { $t, u32 }
        // sh_impl! { $t, u64 }
        sh_impl! { $t, usize }

        // sh_impl! { $t, i8 }
        // sh_impl! { $t, i16 }
        // sh_impl! { $t, i32 }
        // sh_impl! { $t, i64 }
        // sh_impl! { $t, isize }
    )*)
}

mod shift_max {
    #![allow(non_upper_case_globals)]

    pub const  i8: u32 = (1 << 3) - 1;
    pub const i16: u32 = (1 << 4) - 1;
    pub const i32: u32 = (1 << 5) - 1;
    pub const i64: u32 = (1 << 6) - 1;

    pub const  u8: u32 = i8;
    pub const u16: u32 = i16;
    pub const u32: u32 = i32;
    pub const u64: u32 = i64;
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

signed_overflowing_impl! { i8 i16 i32 i64 }
unsigned_overflowing_impl! { u8 u16 u32 u64 }

#[cfg(target_pointer_width = "64")]
impl OverflowingOps for usize {
    #[inline(always)]
    fn overflowing_add(self, rhs: usize) -> (usize, bool) {
        unsafe {
            add_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: usize) -> (usize, bool) {
        unsafe {
            sub_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: usize) -> (usize, bool) {
        unsafe {
            mul_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_div(self, rhs: usize) -> (usize, bool) {
        let (r, f) = (self as u64).overflowing_div(rhs as u64);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_rem(self, rhs: usize) -> (usize, bool) {
        let (r, f) = (self as u64).overflowing_rem(rhs as u64);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_neg(self) -> (usize, bool) {
        let (r, f) = (self as u64).overflowing_neg();
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_shl(self, rhs: u32) -> (usize, bool) {
        let (r, f) = (self as u64).overflowing_shl(rhs);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_shr(self, rhs: u32) -> (usize, bool) {
        let (r, f) = (self as u64).overflowing_shr(rhs);
        (r as usize, f)
    }
}

#[cfg(target_pointer_width = "32")]
impl OverflowingOps for usize {
    #[inline(always)]
    fn overflowing_add(self, rhs: usize) -> (usize, bool) {
        unsafe {
            add_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: usize) -> (usize, bool) {
        unsafe {
            sub_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: usize) -> (usize, bool) {
        unsafe {
            mul_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_div(self, rhs: usize) -> (usize, bool) {
        let (r, f) = (self as u32).overflowing_div(rhs as u32);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_rem(self, rhs: usize) -> (usize, bool) {
        let (r, f) = (self as u32).overflowing_rem(rhs as u32);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_neg(self) -> (usize, bool) {
        let (r, f) = (self as u32).overflowing_neg();
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_shl(self, rhs: u32) -> (usize, bool) {
        let (r, f) = (self as u32).overflowing_shl(rhs);
        (r as usize, f)
    }
    #[inline(always)]
    fn overflowing_shr(self, rhs: u32) -> (usize, bool) {
        let (r, f) = (self as u32).overflowing_shr(rhs);
        (r as usize, f)
    }
}

#[cfg(target_pointer_width = "64")]
impl OverflowingOps for isize {
    #[inline(always)]
    fn overflowing_add(self, rhs: isize) -> (isize, bool) {
        unsafe {
            add_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: isize) -> (isize, bool) {
        unsafe {
            sub_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: isize) -> (isize, bool) {
        unsafe {
            mul_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_div(self, rhs: isize) -> (isize, bool) {
        let (r, f) = (self as i64).overflowing_div(rhs as i64);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_rem(self, rhs: isize) -> (isize, bool) {
        let (r, f) = (self as i64).overflowing_rem(rhs as i64);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_neg(self) -> (isize, bool) {
        let (r, f) = (self as i64).overflowing_neg();
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_shl(self, rhs: u32) -> (isize, bool) {
        let (r, f) = (self as i64).overflowing_shl(rhs);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_shr(self, rhs: u32) -> (isize, bool) {
        let (r, f) = (self as i64).overflowing_shr(rhs);
        (r as isize, f)
    }
}

#[cfg(target_pointer_width = "32")]
impl OverflowingOps for isize {
    #[inline(always)]
    fn overflowing_add(self, rhs: isize) -> (isize, bool) {
        unsafe {
            add_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: isize) -> (isize, bool) {
        unsafe {
            sub_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: isize) -> (isize, bool) {
        unsafe {
            mul_with_overflow(self, rhs)
        }
    }
    #[inline(always)]
    fn overflowing_div(self, rhs: isize) -> (isize, bool) {
        let (r, f) = (self as i32).overflowing_div(rhs as i32);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_rem(self, rhs: isize) -> (isize, bool) {
        let (r, f) = (self as i32).overflowing_rem(rhs as i32);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_neg(self) -> (isize, bool) {
        let (r, f) = (self as i32).overflowing_neg();
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_shl(self, rhs: u32) -> (isize, bool) {
        let (r, f) = (self as i32).overflowing_shl(rhs);
        (r as isize, f)
    }
    #[inline(always)]
    fn overflowing_shr(self, rhs: u32) -> (isize, bool) {
        let (r, f) = (self as i32).overflowing_shr(rhs);
        (r as isize, f)
    }
}
