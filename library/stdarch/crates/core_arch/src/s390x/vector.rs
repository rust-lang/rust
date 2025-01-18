//! s390x vector intrinsics.
//!
//! For more info see the [Reference Summary] or the online [IBM docs].
//!
//! [Reference Summary]: https://www.ibm.com/support/pages/sites/default/files/2021-05/SA22-7871-10.pdf
//! [IBM docs]: https://www.ibm.com/docs/en/zos/2.4.0?topic=support-vector-built-in-functions

#![allow(non_camel_case_types)]

use crate::{core_arch::simd::*, intrinsics::simd::*, mem::transmute};

#[cfg(test)]
use stdarch_test::assert_instr;

use super::macros::*;

types! {
    #![unstable(feature = "stdarch_s390x", issue = "135681")]

    /// s390x-specific 128-bit wide vector of sixteen packed `i8`
    pub struct vector_signed_char(16 x i8);
    /// s390x-specific 128-bit wide vector of sixteen packed `u8`
    pub struct vector_unsigned_char(16 x u8);
    /// s390x-specific 128-bit wide vector mask of sixteen packed elements
    pub struct vector_bool_char(16 x i8);

    /// s390x-specific 128-bit wide vector of eight packed `i16`
    pub struct vector_signed_short(8 x i16);
    /// s390x-specific 128-bit wide vector of eight packed `u16`
    pub struct vector_unsigned_short(8 x u16);
    /// s390x-specific 128-bit wide vector mask of eight packed elements
    pub struct vector_bool_short(8 x i16);

    /// s390x-specific 128-bit wide vector of four packed `i32`
    pub struct vector_signed_int(4 x i32);
    /// s390x-specific 128-bit wide vector of four packed `u32`
    pub struct vector_unsigned_int(4 x u32);
    /// s390x-specific 128-bit wide vector mask of four packed elements
    pub struct vector_bool_int(4 x i32);

    /// s390x-specific 128-bit wide vector of two packed `i64`
    pub struct vector_signed_long_long(2 x i64);
    /// s390x-specific 128-bit wide vector of two packed `u64`
    pub struct vector_unsigned_long_long(2 x u64);
    /// s390x-specific 128-bit wide vector mask of two packed elements
    pub struct vector_bool_long_long(2 x i64);

    /// s390x-specific 128-bit wide vector of four packed `f32`
    pub struct vector_float(4 x f32);
    /// s390x-specific 128-bit wide vector of two packed `f64`
    pub struct vector_double(2 x f64);
}

#[allow(improper_ctypes)]
extern "C" {}

impl_from! { i8x16, u8x16,  i16x8, u16x8, i32x4, u32x4, i64x2, u64x2, f32x4, f64x2 }

impl_neg! { i8x16 : 0 }
impl_neg! { i16x8 : 0 }
impl_neg! { i32x4 : 0 }
impl_neg! { i64x2 : 0 }
impl_neg! { f32x4 : 0f32 }
impl_neg! { f64x2 : 0f64 }

#[macro_use]
mod sealed {
    use super::*;

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAdd<Other> {
        type Result;
        unsafe fn vec_add(self, other: Other) -> Self::Result;
    }

    macro_rules! impl_add {
        ($name:ident, $a:ty, $instr:ident) => {
            impl_add!($name, $a, $a, $a, $instr);
        };
        ($name:ident, $a:ty, $b:ty, $c:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $b) -> $c {
                transmute(simd_add(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorAdd<$b> for $a {
                type Result = $c;

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_add(self, other: $b) -> Self::Result {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_add {
        use super::*;

        impl_add!(va_sc, vector_signed_char, vab);
        impl_add!(va_uc, vector_unsigned_char, vab);
        impl_add!(va_sh, vector_signed_short, vah);
        impl_add!(va_uh, vector_unsigned_short, vah);
        impl_add!(va_sf, vector_signed_int, vaf);
        impl_add!(va_uf, vector_unsigned_int, vaf);
        impl_add!(va_sg, vector_signed_long_long, vag);
        impl_add!(va_ug, vector_unsigned_long_long, vag);

        impl_add!(va_sc_bc, vector_signed_char, vector_bool_char, vector_signed_char, vab);
        impl_add!(va_uc_bc, vector_unsigned_char, vector_bool_char, vector_unsigned_char, vab);
        impl_add!(va_sh_bh, vector_signed_short, vector_bool_short, vector_signed_short, vah);
        impl_add!(va_uh_bh, vector_unsigned_short, vector_bool_short, vector_unsigned_short, vah);
        impl_add!(va_sf_bf, vector_signed_int, vector_bool_int, vector_signed_int, vaf);
        impl_add!(va_uf_bf, vector_unsigned_int, vector_bool_int, vector_unsigned_int, vaf);
        impl_add!(va_sg_bg, vector_signed_long_long, vector_bool_long_long, vector_signed_long_long, vag);
        impl_add!(va_ug_bg, vector_unsigned_long_long, vector_bool_long_long, vector_unsigned_long_long, vag);

        impl_add!(va_bc_sc, vector_bool_char, vector_signed_char, vector_signed_char, vab);
        impl_add!(va_bc_uc, vector_bool_char, vector_unsigned_char, vector_unsigned_char, vab);
        impl_add!(va_bh_sh, vector_bool_short, vector_signed_short, vector_signed_short, vah);
        impl_add!(va_bh_uh, vector_bool_short, vector_unsigned_short, vector_unsigned_short, vah);
        impl_add!(va_bf_sf, vector_bool_int, vector_signed_int, vector_signed_int, vaf);
        impl_add!(va_bf_uf, vector_bool_int, vector_unsigned_int, vector_unsigned_int, vaf);
        impl_add!(va_bg_sg, vector_bool_long_long, vector_signed_long_long, vector_signed_long_long, vag);
        impl_add!(va_bg_ug, vector_bool_long_long, vector_unsigned_long_long, vector_unsigned_long_long, vag);

        impl_add!(va_double, vector_double, vfadb);

        #[inline]
        #[target_feature(enable = "vector")]
        // FIXME: "vfasb" is part of vector enhancements 1, add a test for it when possible
        // #[cfg_attr(test, assert_instr(vfasb))]
        pub unsafe fn va_float(a: vector_float, b: vector_float) -> vector_float {
            transmute(simd_add(a, b))
        }

        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl VectorAdd<Self> for vector_float {
            type Result = Self;

            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn vec_add(self, other: Self) -> Self::Result {
                simd_add(self, other)
            }
        }
    }
}

/// Vector pointwise addition.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_add<T, U>(a: T, b: U) -> <T as sealed::VectorAdd<U>>::Result
where
    T: sealed::VectorAdd<U>,
{
    a.vec_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "vector")]
    unsafe fn vec_add_i32x4_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        let y = i32x4::new(4, 3, 2, 1);
        let x: vector_signed_int = transmute(x);
        let y: vector_signed_int = transmute(y);
        let z = vec_add(x, y);
        assert_eq!(i32x4::splat(5), transmute(z));
    }
}
