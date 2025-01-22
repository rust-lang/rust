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
#[rustfmt::skip]
unsafe extern "unadjusted" {
    #[link_name = "llvm.smax.v16i8"] fn vmxb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.smax.v8i16"] fn vmxh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.smax.v4i32"] fn vmxf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.smax.v2i64"] fn vmxg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.umax.v16i8"] fn vmxlb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.umax.v8i16"] fn vmxlh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.umax.v4i32"] fn vmxlf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.umax.v2i64"] fn vmxlg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.smin.v16i8"] fn vmnb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.smin.v8i16"] fn vmnh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.smin.v4i32"] fn vmnf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.smin.v2i64"] fn vmng(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.umin.v16i8"] fn vmnlb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.umin.v8i16"] fn vmnlh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.umin.v4i32"] fn vmnlf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.umin.v2i64"] fn vmnlg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vfisb"] fn vfisb(a: vector_float, b: i32, c: i32) -> vector_float;
    #[link_name = "llvm.s390.vfidb"] fn vfidb(a: vector_double, b: i32, c: i32) -> vector_double;

}

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

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMax<Other> {
        type Result;
        unsafe fn vec_max(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vmxsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vmxb, vmxb] }
    test_impl! { vec_vmxsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vmxh, vmxh] }
    test_impl! { vec_vmxsf (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vmxf, vmxf] }
    test_impl! { vec_vmxsg (a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long [vmxg, vmxg] }

    test_impl! { vec_vmxslb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vmxlb, vmxlb] }
    test_impl! { vec_vmxslh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vmxlh, vmxlh] }
    test_impl! { vec_vmxslf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vmxlf, vmxlf] }
    test_impl! { vec_vmxslg (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vmxlg, vmxlg] }

    impl_vec_trait! { [VectorMax vec_max] ~(vmxlb, vmxb, vmxlh, vmxh, vmxlf, vmxf, vmxlg, vmxg) }

    // FIXME(vector-enhancements-1) test for the `vfmaxsb` etc. instruction
    test_impl! { vec_vfmaxsb (a: vector_float, b: vector_float) -> vector_float [simd_fmax, _] }
    test_impl! { vec_vfmaxdb (a: vector_double, b: vector_double) -> vector_double [simd_fmax, _] }

    impl_vec_trait!([VectorMax vec_max] vec_vfmaxsb (vector_float, vector_float) -> vector_float);
    impl_vec_trait!([VectorMax vec_max] vec_vfmaxdb (vector_double, vector_double) -> vector_double);

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMin<Other> {
        type Result;
        unsafe fn vec_min(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vmnsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vmnb, vmnb] }
    test_impl! { vec_vmnsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vmnh, vmnh] }
    test_impl! { vec_vmnsf (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vmnf, vmnf] }
    test_impl! { vec_vmnsg (a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long [vmng, vmng] }

    test_impl! { vec_vmnslb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vmnlb, vmnlb] }
    test_impl! { vec_vmnslh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vmnlh, vmnlh] }
    test_impl! { vec_vmnslf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vmnlf, vmnlf] }
    test_impl! { vec_vmnslg (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vmnlg, vmnlg] }

    impl_vec_trait! { [VectorMin vec_min] ~(vmxlb, vmxb, vmxlh, vmxh, vmxlf, vmxf, vmxlg, vmxg) }

    // FIXME(vector-enhancements-1) test for the `vfminsb` etc. instruction
    test_impl! { vec_vfminsb (a: vector_float, b: vector_float) -> vector_float [simd_fmin, _] }
    test_impl! { vec_vfmindb (a: vector_double, b: vector_double) -> vector_double [simd_fmin, _] }

    impl_vec_trait!([VectorMin vec_min] vec_vfminsb (vector_float, vector_float) -> vector_float);
    impl_vec_trait!([VectorMin vec_min] vec_vfmindb (vector_double, vector_double) -> vector_double);

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAbs {
        unsafe fn vec_abs(self) -> Self;
    }

    macro_rules! impl_abs {
        ($name:ident, $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $name(v: s_t_l!($ty)) -> s_t_l!($ty) {
                v.vec_max(-v)
            }

            impl_vec_trait! { [VectorAbs vec_abs] $name (s_t_l!($ty)) }
        };
    }

    impl_abs! { vec_abs_i8, i8x16 }
    impl_abs! { vec_abs_i16, i16x8 }
    impl_abs! { vec_abs_i32, i32x4 }
    impl_abs! { vec_abs_i64, i64x2 }

    // FIXME(vector-enhancements-1)
    test_impl! { vec_abs_f32 (v: vector_float) -> vector_float [ simd_fabs, _ ] }
    test_impl! { vec_abs_f64 (v: vector_double) -> vector_double [ simd_fabs, vflpdb ] }

    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f32 (vector_float) }
    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSplats<Output> {
        unsafe fn vec_splats(self) -> Output;
    }

    macro_rules! impl_vec_splats {
        ($(($fn:ident ($ty:ty, $shortty:tt) $instr:ident)),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($instr))]
                pub unsafe fn $fn(v: $ty) -> s_t_l!($shortty) {
                    transmute($shortty::splat(v))
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSplats<s_t_l!($shortty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_splats(self) -> s_t_l!($shortty) {
                        $fn (self)
                    }
                }
            )*
        }
    }

    impl_vec_splats! {
        (vec_splats_u8 (u8, u8x16) vrepb),
        (vec_splats_i8 (i8, i8x16) vrepb),
        (vec_splats_u16 (u16, u16x8) vreph),
        (vec_splats_i16 (i16, i16x8) vreph),
        (vec_splats_u32 (u32, u32x4) vrepf),
        (vec_splats_i32 (i32, i32x4) vrepf),
        (vec_splats_u64 (u64, u64x2) vlvgp),
        (vec_splats_i64 (i64, i64x2) vlvgp),
        (vec_splats_f32 (f32, f32x4) vrepf),
        (vec_splats_f64 (f64, f64x2) vrepg)
    }

    macro_rules! impl_bool_vec_splats {
        ($(($ty:ty, $shortty:tt, $boolty:ty)),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSplats<$boolty> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_splats(self) -> $boolty {
                        transmute($shortty::splat(self))
                    }
                }
            )*
        }
    }

    impl_bool_vec_splats! {
        (u8, u8x16, vector_bool_char),
        (i8, i8x16, vector_bool_char),
        (u16, u16x8, vector_bool_short),
        (i16, i16x8, vector_bool_short),
        (u32, u32x4, vector_bool_int),
        (i32, i32x4, vector_bool_int),
        (u64, u64x2, vector_bool_long_long),
        (i64, i64x2, vector_bool_long_long)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait CountBits {
        type Result;

        unsafe fn vec_cntlz(self) -> Self::Result;
        unsafe fn vec_cnttz(self) -> Self::Result;
        unsafe fn vec_popcnt(self) -> Self::Result;
    }

    macro_rules! impl_count_bits {
        ($ty:tt) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl CountBits for $ty {
                type Result = t_u!($ty);

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_cntlz(self) -> Self::Result {
                    transmute(simd_ctlz(self))
                }

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_cnttz(self) -> Self::Result {
                    transmute(simd_cttz(self))
                }

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_popcnt(self) -> Self::Result {
                    transmute(simd_ctpop(self))
                }
            }
        };
    }

    impl_count_bits!(vector_signed_char);
    impl_count_bits!(vector_unsigned_char);
    impl_count_bits!(vector_signed_short);
    impl_count_bits!(vector_unsigned_short);
    impl_count_bits!(vector_signed_int);
    impl_count_bits!(vector_unsigned_int);
    impl_count_bits!(vector_signed_long_long);
    impl_count_bits!(vector_unsigned_long_long);

    test_impl! { vec_clzb_signed +(a: vector_signed_char) -> vector_unsigned_char [simd_ctlz, vclzb] }
    test_impl! { vec_clzh_signed +(a: vector_signed_short) -> vector_unsigned_short [simd_ctlz, vclzh] }
    test_impl! { vec_clzf_signed +(a: vector_signed_int) -> vector_unsigned_int [simd_ctlz, vclzf] }
    test_impl! { vec_clzg_signed +(a: vector_signed_long_long) -> vector_unsigned_long_long [simd_ctlz, vclzg] }

    test_impl! { vec_clzb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_ctlz, vclzb] }
    test_impl! { vec_clzh_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_ctlz, vclzh] }
    test_impl! { vec_clzf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_ctlz, vclzf] }
    test_impl! { vec_clzg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_ctlz, vclzg] }

    test_impl! { vec_ctzb_signed +(a: vector_signed_char) -> vector_unsigned_char [simd_cttz, vctzb] }
    test_impl! { vec_ctzh_signed +(a: vector_signed_short) -> vector_unsigned_short [simd_cttz, vctzh] }
    test_impl! { vec_ctzf_signed +(a: vector_signed_int) -> vector_unsigned_int [simd_cttz, vctzf] }
    test_impl! { vec_ctzg_signed +(a: vector_signed_long_long) -> vector_unsigned_long_long [simd_cttz, vctzg] }

    test_impl! { vec_ctzb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_cttz, vctzb] }
    test_impl! { vec_ctzh_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_cttz, vctzh] }
    test_impl! { vec_ctzf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_cttz, vctzf] }
    test_impl! { vec_ctzg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_cttz, vctzg] }

    // FIXME(vector-enhancements-1) other integer types are emulated, but get their own
    // instructions in later facilities. Add tests when possible.
    test_impl! { vec_popcnt_signed +(a: vector_signed_char) -> vector_signed_char [simd_ctpop, vpopctb] }
    test_impl! { vec_popcnt_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_ctpop, vpopctb] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAnd<Other> {
        type Result;
        unsafe fn vec_and(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAnd vec_and] ~(simd_and) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorOr<Other> {
        type Result;
        unsafe fn vec_or(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorOr vec_or] ~(simd_or) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorXor<Other> {
        type Result;
        unsafe fn vec_xor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorXor vec_xor] ~(simd_xor) }

    #[inline]
    #[target_feature(enable = "vector")]
    // FIXME(vector-enhancements-1) #[cfg_attr(test, assert_instr(vno))]
    unsafe fn nor(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_or(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNor<Other> {
        type Result;
        unsafe fn vec_nor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorNor vec_nor]+ 2c (nor) }

    #[inline]
    #[target_feature(enable = "vector")]
    // FIXME(vector-enhancements-1) #[cfg_attr(test, assert_instr(vnn))]
    unsafe fn nand(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_and(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNand<Other> {
        type Result;
        unsafe fn vec_nand(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorNand vec_nand]+ 2c (nand) }

    #[inline]
    #[target_feature(enable = "vector")]
    // FIXME(vector-enhancements-1) #[cfg_attr(test, assert_instr(vnx))]
    unsafe fn eqv(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_xor(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorEqv<Other> {
        type Result;
        unsafe fn vec_eqv(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorEqv vec_eqv]+ 2c (eqv) }

    #[inline]
    #[target_feature(enable = "vector")]
    // FIXME(vector-enhancements-1) #[cfg_attr(test, assert_instr(vnc))]
    unsafe fn andc(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a = transmute(a);
        let b = transmute(b);
        transmute(simd_and(simd_xor(u8x16::splat(0xff), b), a))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAndc<Other> {
        type Result;
        unsafe fn vec_andc(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAndc vec_andc]+ 2c (andc) }

    #[inline]
    #[target_feature(enable = "vector")]
    // FIXME(vector-enhancements-1) #[cfg_attr(test, assert_instr(voc))]
    unsafe fn orc(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a = transmute(a);
        let b = transmute(b);
        transmute(simd_or(simd_xor(u8x16::splat(0xff), b), a))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorOrc<Other> {
        type Result;
        unsafe fn vec_orc(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorOrc vec_orc]+ 2c (orc) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRound: Sized {
        unsafe fn vec_round_impl<const N: i32, const MODE: i32>(self) -> Self;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_roundc(self) -> Self {
            self.vec_round_impl::<4, 0>()
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_round(self) -> Self {
            // NOTE: simd_round resoles ties by rounding away from zero,
            // while the vec_round function rounds towards zero
            self.vec_round_impl::<4, 4>()
        }

        // NOTE: vec_roundz (vec_round_impl::<4, 5>) is the same as vec_trunc
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_trunc(self) -> Self {
            simd_trunc(self)
        }

        // NOTE: vec_roundp (vec_round_impl::<4, 6>) is the same as vec_ceil
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_ceil(self) -> Self {
            simd_ceil(self)
        }

        // NOTE: vec_roundm (vec_round_impl::<4, 7>) is the same as vec_floor
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_floor(self) -> Self {
            simd_floor(self)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_rint(self) -> Self {
            self.vec_round_impl::<0, 0>()
        }
    }

    // FIXME(vector-enhancements-1) apply the right target feature to all methods
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRound for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_round_impl<const N: i32, const MODE: i32>(self) -> Self {
            vfisb(self, N, MODE)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRound for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_round_impl<const N: i32, const MODE: i32>(self) -> Self {
            vfidb(self, N, MODE)
        }
    }
}

/// Vector element-wise addition.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_add<T, U>(a: T, b: U) -> <T as sealed::VectorAdd<U>>::Result
where
    T: sealed::VectorAdd<U>,
{
    a.vec_add(b)
}

/// Vector element-wise subtraction.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sub<T, U>(a: T, b: U) -> <T as sealed::VectorSub<U>>::Result
where
    T: sealed::VectorSub<U>,
{
    a.vec_sub(b)
}

/// Vector element-wise multiplication.
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

/// Vector Count Leading Zeros
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cntlz<T>(a: T) -> <T as sealed::CountBits>::Result
where
    T: sealed::CountBits,
{
    a.vec_cntlz()
}

/// Vector Count Trailing Zeros
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cnttz<T>(a: T) -> <T as sealed::CountBits>::Result
where
    T: sealed::CountBits,
{
    a.vec_cnttz()
}

/// Vector Population Count
///
/// Computes the population count (number of set bits) in each element of the input.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_popcnt<T>(a: T) -> <T as sealed::CountBits>::Result
where
    T: sealed::CountBits,
{
    a.vec_popcnt()
}

/// Vector element-wise maximum.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_max<T, U>(a: T, b: U) -> <T as sealed::VectorMax<U>>::Result
where
    T: sealed::VectorMax<U>,
{
    a.vec_max(b)
}

/// Vector element-wise minimum.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_min<T, U>(a: T, b: U) -> <T as sealed::VectorMin<U>>::Result
where
    T: sealed::VectorMin<U>,
{
    a.vec_min(b)
}

/// Vector abs.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_abs<T>(a: T) -> T
where
    T: sealed::VectorAbs,
{
    a.vec_abs()
}

/// Vector splats.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_splats<T, U>(a: T) -> U
where
    T: sealed::VectorSplats<U>,
{
    a.vec_splats()
}

/// Vector and
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_and<T, U>(a: T, b: U) -> <T as sealed::VectorAnd<U>>::Result
where
    T: sealed::VectorAnd<U>,
{
    a.vec_and(b)
}

/// Vector or
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_or<T, U>(a: T, b: U) -> <T as sealed::VectorOr<U>>::Result
where
    T: sealed::VectorOr<U>,
{
    a.vec_or(b)
}

/// Vector xor
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_xor<T, U>(a: T, b: U) -> <T as sealed::VectorXor<U>>::Result
where
    T: sealed::VectorXor<U>,
{
    a.vec_xor(b)
}

/// Vector nor
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nor<T, U>(a: T, b: U) -> <T as sealed::VectorNor<U>>::Result
where
    T: sealed::VectorNor<U>,
{
    a.vec_nor(b)
}

/// Vector nand
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nand<T, U>(a: T, b: U) -> <T as sealed::VectorNand<U>>::Result
where
    T: sealed::VectorNand<U>,
{
    a.vec_nand(b)
}

/// Vector xnor
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_eqv<T, U>(a: T, b: U) -> <T as sealed::VectorEqv<U>>::Result
where
    T: sealed::VectorEqv<U>,
{
    a.vec_eqv(b)
}

/// Vector andc.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_andc<T, U>(a: T, b: U) -> <T as sealed::VectorAndc<U>>::Result
where
    T: sealed::VectorAndc<U>,
{
    a.vec_andc(b)
}

/// Vector OR with Complement
///
/// ## Purpose
/// Performs a bitwise OR of the first vector with the bitwise-complemented second vector.
///
/// ## Result value
/// r is the bitwise OR of a and the bitwise complement of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_orc<T, U>(a: T, b: U) -> <T as sealed::VectorOrc<U>>::Result
where
    T: sealed::VectorOrc<U>,
{
    a.vec_orc(b)
}

/// Vector floor.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_floor<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_floor()
}

/// Vector ceil.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_ceil<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_ceil()
}

/// Returns a vector containing the truncated values of the corresponding elements of the given vector.
/// Each element of the result contains the value of the corresponding element of a, truncated to an integral value.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_trunc<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_trunc()
}

/// Vector round, resolves ties by rounding towards zero.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_round<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_round()
}

/// Returns a vector by using the current rounding mode to round every
/// floating-point element in the given vector to integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundc<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_roundc()
}

/// Returns a vector containing the largest representable floating-point integral values less
/// than or equal to the values of the corresponding elements of the given vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundm<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    // the IBM docs note
    //
    // > vec_roundm provides the same functionality as vec_floor, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_floor` also does not trigger that exception, so both are equivalent
    a.vec_floor()
}

/// Returns a vector containing the smallest representable floating-point integral values greater
/// than or equal to the values of the corresponding elements of the given vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundp<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    // the IBM docs note
    //
    // > vec_roundp provides the same functionality as vec_ceil, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_ceil` also does not trigger that exception, so both are equivalent
    a.vec_ceil()
}

/// Returns a vector containing the truncated values of the corresponding elements of the given vector.
/// Each element of the result contains the value of the corresponding element of a, truncated to an integral value.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundz<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    // the IBM docs note
    //
    // > vec_roundz provides the same functionality as vec_trunc, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_trunc` also does not trigger that exception, so both are equivalent
    a.vec_trunc()
}

/// Returns a vector by using the current rounding mode to round every floating-point element in the given vector to integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rint<T>(a: T) -> T
where
    T: sealed::VectorRound,
{
    a.vec_rint()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    macro_rules! test_vec_1 {
        { $name: ident, $fn:ident, f32x4, [$($a:expr),+], ~[$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: vector_float = transmute(f32x4::new($($a),+));

                let d: vector_float = transmute(f32x4::new($($d),+));
                let r = transmute(vec_cmple(vec_abs(vec_sub($fn(a), d)), vec_splats(f32::EPSILON)));
                let e = m32x4::new(true, true, true, true);
                assert_eq!(e, r);
            }
        };
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($d:expr),+] } => {
            test_vec_1! { $name, $fn, $ty -> $ty, [$($a),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a));
                assert_eq!(d, r);
            }
        }
    }

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

    macro_rules! test_vec_abs {
        { $name: ident, $ty: ident, $a: expr, $d: expr } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = vec_splats($a);
                let a: s_t_l!($ty) = vec_abs(a);
                let d = $ty::splat($d);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_abs! { test_vec_abs_i8, i8x16, -42i8, 42i8 }
    test_vec_abs! { test_vec_abs_i16, i16x8, -42i16, 42i16 }
    test_vec_abs! { test_vec_abs_i32, i32x4, -42i32, 42i32 }
    test_vec_abs! { test_vec_abs_i64, i64x2, -42i64, 42i64 }
    test_vec_abs! { test_vec_abs_f32, f32x4, -42f32, 42f32 }
    test_vec_abs! { test_vec_abs_f64, f64x2, -42f64, 42f64 }

    test_vec_2! { test_vec_andc, vec_andc, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b10000000],
    [0b11001100, 0b00001100, 0b11000000, 0b01001100] }

    test_vec_2! { test_vec_and, vec_and, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b00000000, 0b11000000, 0b00001100, 0b00000000] }

    test_vec_2! { test_vec_nand, vec_nand, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b00000000, !0b11000000, !0b00001100, !0b00000000] }

    test_vec_2! { test_vec_orc, vec_orc, u32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11001100 | !0b00110011, 0b11001100 | !0b11110011, 0b11001100 | !0b00001100, 0b11001100 | !0b00000000] }

    test_vec_2! { test_vec_or, vec_or, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11111111, 0b11111111, 0b11001100, 0b11001100] }

    test_vec_2! { test_vec_nor, vec_nor, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b11111111, !0b11111111, !0b11001100, !0b11001100] }

    test_vec_2! { test_vec_xor, vec_xor, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11111111, 0b00111111, 0b11000000, 0b11001100] }

    test_vec_2! { test_vec_eqv, vec_eqv, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b11111111, !0b00111111, !0b11000000, !0b11001100] }

    test_vec_1! { test_vec_floor_f32, vec_floor, f32x4,
        [1.1, 1.9, -0.5, -0.9],
        [1.0, 1.0, -1.0, -1.0]
    }

    test_vec_1! { test_vec_floor_f64_1, vec_floor, f64x2,
        [1.1, 1.9],
        [1.0, 1.0]
    }
    test_vec_1! { test_vec_floor_f64_2, vec_floor, f64x2,
        [-0.5, -0.9],
        [-1.0, -1.0]
    }

    test_vec_1! { test_vec_ceil_f32, vec_ceil, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [1.0, 1.0, 1.0, 1.0]
    }
    test_vec_1! { test_vec_ceil_f64_1, vec_ceil, f64x2,
        [0.1, 0.5],
        [1.0, 1.0]
    }
    test_vec_1! { test_vec_ceil_f64_2, vec_ceil, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    // FIXME(vector-enhancements-1)
    //    test_vec_1! { test_vec_round_f32, vec_round, f32x4,
    //        [],
    //        []
    //    }
    test_vec_1! { test_vec_round_f64_1, vec_round, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_round_f64_2, vec_round, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    // FIXME(vector-enhancements-1)
    //    test_vec_1! { test_vec_roundc_f32, vec_roundc, f32x4,
    //        [],
    //        []
    //    }
    test_vec_1! { test_vec_roundc_f64_1, vec_roundc, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_roundc_f64_2, vec_roundc, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    // FIXME(vector-enhancements-1)
    //    test_vec_1! { test_vec_rint_f32, vec_rint, f32x4,
    //        [],
    //        []
    //    }
    test_vec_1! { test_vec_rint_f64_1, vec_rint, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_rint_f64_2, vec_rint, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }
}
