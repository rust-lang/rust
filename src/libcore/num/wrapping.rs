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
