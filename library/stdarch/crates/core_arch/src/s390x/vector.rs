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
                va_float(self, other)
            }
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSub<Other> {
        type Result;
        unsafe fn vec_sub(self, other: Other) -> Self::Result;
    }

    macro_rules! impl_sub {
        ($name:ident, $a:ty, $instr:ident) => {
            impl_sub!($name, $a, $a, $a, $instr);
        };
        ($name:ident, $a:ty, $b:ty, $c:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $b) -> $c {
                transmute(simd_sub(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorSub<$b> for $a {
                type Result = $c;

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_sub(self, other: $b) -> Self::Result {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_sub {
        use super::*;

        impl_sub!(vs_sc, vector_signed_char, vsb);
        impl_sub!(vs_uc, vector_unsigned_char, vsb);
        impl_sub!(vs_sh, vector_signed_short, vsh);
        impl_sub!(vs_uh, vector_unsigned_short, vsh);
        impl_sub!(vs_sf, vector_signed_int, vsf);
        impl_sub!(vs_uf, vector_unsigned_int, vsf);
        impl_sub!(vs_sg, vector_signed_long_long, vsg);
        impl_sub!(vs_ug, vector_unsigned_long_long, vsg);

        impl_sub!(vs_sc_bc, vector_signed_char, vector_bool_char, vector_signed_char, vsb);
        impl_sub!(vs_uc_bc, vector_unsigned_char, vector_bool_char, vector_unsigned_char, vsb);
        impl_sub!(vs_sh_bh, vector_signed_short, vector_bool_short, vector_signed_short, vsh);
        impl_sub!(vs_uh_bh, vector_unsigned_short, vector_bool_short, vector_unsigned_short, vsh);
        impl_sub!(vs_sf_bf, vector_signed_int, vector_bool_int, vector_signed_int, vsf);
        impl_sub!(vs_uf_bf, vector_unsigned_int, vector_bool_int, vector_unsigned_int, vsf);
        impl_sub!(vs_sg_bg, vector_signed_long_long, vector_bool_long_long, vector_signed_long_long, vsg);
        impl_sub!(vs_ug_bg, vector_unsigned_long_long, vector_bool_long_long, vector_unsigned_long_long, vsg);

        impl_sub!(vs_bc_sc, vector_bool_char, vector_signed_char, vector_signed_char, vsb);
        impl_sub!(vs_bc_uc, vector_bool_char, vector_unsigned_char, vector_unsigned_char, vsb);
        impl_sub!(vs_bh_sh, vector_bool_short, vector_signed_short, vector_signed_short, vsh);
        impl_sub!(vs_bh_uh, vector_bool_short, vector_unsigned_short, vector_unsigned_short, vsh);
        impl_sub!(vs_bf_sf, vector_bool_int, vector_signed_int, vector_signed_int, vsf);
        impl_sub!(vs_bf_uf, vector_bool_int, vector_unsigned_int, vector_unsigned_int, vsf);
        impl_sub!(vs_bg_sg, vector_bool_long_long, vector_signed_long_long, vector_signed_long_long, vsg);
        impl_sub!(vs_bg_ug, vector_bool_long_long, vector_unsigned_long_long, vector_unsigned_long_long, vsg);

        impl_sub!(vs_double, vector_double, vfsdb);

        #[inline]
        #[target_feature(enable = "vector")]
        // FIXME: "vfssb" is part of vector enhancements 1, add a test for it when possible
        // #[cfg_attr(test, assert_instr(vfasb))]
        pub unsafe fn vs_float(a: vector_float, b: vector_float) -> vector_float {
            transmute(simd_sub(a, b))
        }

        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl VectorSub<Self> for vector_float {
            type Result = Self;

            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn vec_sub(self, other: Self) -> Self::Result {
                vs_float(self, other)
            }
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMul {
        unsafe fn vec_mul(self, b: Self) -> Self;
    }

    macro_rules! impl_mul {
        ($name:ident, $a:ty, std_simd) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorMul for $a {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_mul(self, other: Self) -> Self {
                    transmute(simd_mul(transmute(self), other))
                }
            }
        };
        ($name:ident, $a:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $a) -> $a {
                transmute(simd_mul(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorMul for $a {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_mul(self, other: Self) -> Self {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_mul {
        use super::*;

        impl_mul!(vml_sc, vector_signed_char, vmlb);
        impl_mul!(vml_uc, vector_unsigned_char, vmlb);
        impl_mul!(vml_sh, vector_signed_short, vmlhw);
        impl_mul!(vml_uh, vector_unsigned_short, vmlhw);
        impl_mul!(vml_sf, vector_signed_int, vmlf);
        impl_mul!(vml_uf, vector_unsigned_int, vmlf);
        impl_mul!(vml_sg, vector_signed_long_long, std_simd);
        impl_mul!(vml_ug, vector_unsigned_long_long, std_simd);

        impl_mul!(vml_float, vector_float, std_simd);
        impl_mul!(vml_double, vector_double, vfmdb);
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

/// Vector pointwise subtraction.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sub<T, U>(a: T, b: U) -> <T as sealed::VectorSub<U>>::Result
where
    T: sealed::VectorSub<U>,
{
    a.vec_sub(b)
}

/// Vector Multiply
///
/// ## Purpose
/// Compute the products of corresponding elements of two vectors.
///
/// ## Result value
/// Each element of r receives the product of the corresponding elements of a and b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mul<T>(a: T, b: T) -> T
where
    T: sealed::VectorMul,
{
    a.vec_mul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    macro_rules! test_vec_2 {
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! { $name, $fn, $ty -> $ty, [$($a),+], [$($b),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a, b));
                assert_eq!(d, r);
            }
         };
         { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], $d:expr } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let r : $ty_out = transmute($fn(a, b));
                assert_eq!($d, r);
            }
         }
   }

    #[simd_test(enable = "vector")]
    unsafe fn vec_add_i32x4_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        let y = i32x4::new(4, 3, 2, 1);
        let x: vector_signed_int = transmute(x);
        let y: vector_signed_int = transmute(y);
        let z = vec_add(x, y);
        assert_eq!(i32x4::splat(5), transmute(z));
    }

    macro_rules! test_vec_sub {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_sub, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_sub! { test_vec_sub_f32x4, f32x4,
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, -1.0, -2.0],
    [-3.0, -1.0, 2.0, 4.0] }

    test_vec_sub! { test_vec_sub_f64x2, f64x2,
    [-1.0, 0.0],
    [2.0, 1.0],
    [-3.0, -1.0] }

    test_vec_sub! { test_vec_sub_i64x2, i64x2,
    [-1, 0],
    [2, 1],
    [-3, -1] }

    test_vec_sub! { test_vec_sub_u64x2, u64x2,
    [0, 1],
    [1, 0],
    [u64::MAX, 1] }

    test_vec_sub! { test_vec_sub_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u32x4, u32x4,
    [0, 0, 1, 2],
    [2, 1, 0, 0],
    [4294967294, 4294967295, 1, 2] }

    test_vec_sub! { test_vec_sub_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u16x8, u16x8,
    [0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0],
    [65534, 65535, 1, 2, 65534, 65535, 1, 2] }

    test_vec_sub! { test_vec_sub_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u8x16, u8x16,
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
    [254, 255, 1, 2, 254, 255, 1, 2, 254, 255, 1, 2, 254, 255, 1, 2] }

    macro_rules! test_vec_mul {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_mul, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_mul! { test_vec_mul_f32x4, f32x4,
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, -1.0, -2.0],
    [-2.0, 0.0, -1.0, -4.0] }

    test_vec_mul! { test_vec_mul_f64x2, f64x2,
    [-1.0, 0.0],
    [2.0, 1.0],
    [-2.0, 0.0] }

    test_vec_mul! { test_vec_mul_i64x2, i64x2,
    [i64::MAX, -4],
    [2, 3],
    [i64::MAX.wrapping_mul(2), -12] }

    test_vec_mul! { test_vec_mul_u64x2, u64x2,
    [u64::MAX, 4],
    [2, 3],
    [u64::MAX.wrapping_mul(2), 12] }

    test_vec_mul! { test_vec_mul_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u32x4, u32x4,
    [0, u32::MAX - 1, 1, 2],
    [5, 6, 7, 8],
    [0, 4294967284, 7, 16] }

    test_vec_mul! { test_vec_mul_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-2, 0, -1, -4, -2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u16x8, u16x8,
    [0, u16::MAX - 1, 1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 9, 8, 7, 6],
    [0, 65524, 7, 16, 27, 32, 35, 36] }

    test_vec_mul! { test_vec_mul_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-2, 0, -1, -4, -2, 0, -1, -4, -2, 0, -1, -4, -2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u8x16, u8x16,
    [0, u8::MAX - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4],
    [5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 0, u8::MAX, 1, 2, 3, 4],
    [0, 244, 7, 16, 27, 32, 35, 36, 35, 32, 0, 248, 7, 12, 15, 16] }
}
