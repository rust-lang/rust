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

#[repr(packed)]
struct PackedTuple<T, U> {
    x: T,
    y: U,
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

    #[link_name = "llvm.nearbyint.v4f32"] fn nearbyint_v4f32(a: vector_float) -> vector_float;
    #[link_name = "llvm.nearbyint.v2f64"] fn nearbyint_v2f64(a: vector_double) -> vector_double;

    #[link_name = "llvm.rint.v4f32"] fn rint_v4f32(a: vector_float) -> vector_float;
    #[link_name = "llvm.rint.v2f64"] fn rint_v2f64(a: vector_double) -> vector_double;

    #[link_name = "llvm.roundeven.v4f32"] fn roundeven_v4f32(a: vector_float) -> vector_float;
    #[link_name = "llvm.roundeven.v2f64"] fn roundeven_v2f64(a: vector_double) -> vector_double;

    #[link_name = "llvm.s390.vsra"] fn vsra(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsrl"] fn vsrl(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsl"] fn vsl(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;

    #[link_name = "llvm.s390.vsrab"] fn vsrab(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsrlb"] fn vsrlb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vslb"] fn vslb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;

    #[link_name = "llvm.fshl.v16i8"] fn fshlb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.fshl.v8i16"] fn fshlh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.fshl.v4i32"] fn fshlf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.fshl.v2i64"] fn fshlg(a: vector_unsigned_long_long, b: vector_unsigned_long_long, c: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.verimb"] fn verimb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char, d: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.verimh"] fn verimh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short, d: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.verimf"] fn verimf(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int, d: i32) -> vector_signed_int;
    #[link_name = "llvm.s390.verimg"] fn verimg(a: vector_signed_long_long, b: vector_signed_long_long, c: vector_signed_long_long, d: i32) -> vector_signed_long_long;

    #[link_name = "llvm.s390.vperm"] fn vperm(a: vector_signed_char, b: vector_signed_char, c: vector_unsigned_char) -> vector_signed_char;

    #[link_name = "llvm.s390.vsumb"] fn vsumb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vsumh"] fn vsumh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vsumgh"] fn vsumgh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_long_long;
    #[link_name = "llvm.s390.vsumgf"] fn vsumgf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vsumqf"] fn vsumqf(a: vector_unsigned_int, b: vector_unsigned_int) -> u128;
    #[link_name = "llvm.s390.vsumqg"] fn vsumqg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> u128;

    #[link_name = "llvm.s390.vscbiq"] fn vscbiq(a: u128, b: u128) -> u128;
    #[link_name = "llvm.s390.vsbiq"] fn vsbiq(a: u128, b: u128, c: u128) -> u128;
    #[link_name = "llvm.s390.vsbcbiq"] fn vsbcbiq(a: u128, b: u128, c: u128) -> u128;

    #[link_name = "llvm.s390.vscbib"] fn vscbib(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vscbih"] fn vscbih(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vscbif"] fn vscbif(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vscbig"] fn vscbig(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vfaeb"] fn vfaeb(a: vector_signed_char, b: vector_signed_char, c: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.vfaeh"] fn vfaeh(a: vector_signed_short, b: vector_signed_short, c: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.vfaef"] fn vfaef(a: vector_signed_int, b: vector_signed_int, c: i32) -> vector_signed_int;

    #[link_name = "llvm.s390.vfaezb"] fn vfaezb(a: vector_signed_char, b: vector_signed_char, c: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.vfaezh"] fn vfaezh(a: vector_signed_short, b: vector_signed_short, c: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.vfaezf"] fn vfaezf(a: vector_signed_int, b: vector_signed_int, c: i32) -> vector_signed_int;

    #[link_name = "llvm.s390.vfaebs"] fn vfaebs(a: vector_signed_char, b: vector_signed_char, c: i32) -> PackedTuple<vector_signed_char, i32>;
    #[link_name = "llvm.s390.vfaehs"] fn vfaehs(a: vector_signed_short, b: vector_signed_short, c: i32) -> PackedTuple<vector_signed_short, i32>;
    #[link_name = "llvm.s390.vfaefs"] fn vfaefs(a: vector_signed_int, b: vector_signed_int, c: i32) -> PackedTuple<vector_signed_int, i32>;

    #[link_name = "llvm.s390.vfaezbs"] fn vfaezbs(a: vector_signed_char, b: vector_signed_char, c: i32) -> PackedTuple<vector_signed_char, i32>;
    #[link_name = "llvm.s390.vfaezhs"] fn vfaezhs(a: vector_signed_short, b: vector_signed_short, c: i32) -> PackedTuple<vector_signed_short, i32>;
    #[link_name = "llvm.s390.vfaezfs"] fn vfaezfs(a: vector_signed_int, b: vector_signed_int, c: i32) -> PackedTuple<vector_signed_int, i32>;
}

impl_from! { i8x16, u8x16,  i16x8, u16x8, i32x4, u32x4, i64x2, u64x2, f32x4, f64x2 }

impl_neg! { i8x16 : 0 }
impl_neg! { i16x8 : 0 }
impl_neg! { i32x4 : 0 }
impl_neg! { i64x2 : 0 }
impl_neg! { f32x4 : 0f32 }
impl_neg! { f64x2 : 0f64 }

#[repr(simd)]
struct ShuffleMask<const N: usize>([u32; N]);

impl<const N: usize> ShuffleMask<N> {
    const fn reverse() -> Self {
        let mut index = [0; N];
        let mut i = 0;
        while i < N {
            index[i] = (N - i - 1) as u32;
            i += 1;
        }
        ShuffleMask(index)
    }

    const fn merge_low() -> Self {
        let mut mask = [0; N];
        let mut i = N / 2;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;
            mask[index + 1] = (i + N) as u32;

            i += 1;
            index += 2;
        }
        ShuffleMask(mask)
    }

    const fn merge_high() -> Self {
        let mut mask = [0; N];
        let mut i = 0;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;
            mask[index + 1] = (i + N) as u32;

            i += 1;
            index += 2;
        }
        ShuffleMask(mask)
    }
}

const fn genmask<const MASK: u16>() -> [u8; 16] {
    let mut bits = MASK;
    let mut elements = [0u8; 16];

    let mut i = 0;
    while i < 16 {
        elements[i] = match bits & (1u16 << 15) {
            0 => 0,
            _ => 0xFF,
        };

        bits <<= 1;
        i += 1;
    }

    elements
}

const fn genmasks(bit_width: u32, a: u8, b: u8) -> u64 {
    let bit_width = bit_width as u8;
    let a = a % bit_width;
    let mut b = b % bit_width;
    if a > b {
        b = bit_width - 1;
    }

    // of course these indices start from the left
    let a = (bit_width - 1) - a;
    let b = (bit_width - 1) - b;

    ((1u64.wrapping_shl(a as u32 + 1)) - 1) & !((1u64.wrapping_shl(b as u32)) - 1)
}

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
        #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vfasb))]
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
        #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vfssb))]
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

    test_impl! { vec_vfmaxsb (a: vector_float, b: vector_float) -> vector_float [simd_fmax, "vector-enhancements-1" vfmaxsb ] }
    test_impl! { vec_vfmaxdb (a: vector_double, b: vector_double) -> vector_double [simd_fmax, "vector-enhancements-1" vfmaxdb] }

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

    test_impl! { vec_vfminsb (a: vector_float, b: vector_float) -> vector_float [simd_fmin, "vector-enhancements-1" vfminsb]  }
    test_impl! { vec_vfmindb (a: vector_double, b: vector_double) -> vector_double [simd_fmin, "vector-enhancements-1" vfmindb]  }

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

    test_impl! { vec_abs_f32 (v: vector_float) -> vector_float [ simd_fabs, "vector-enhancements-1" vflpsb ] }
    test_impl! { vec_abs_f64 (v: vector_double) -> vector_double [ simd_fabs, vflpdb ] }

    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f32 (vector_float) }
    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNabs {
        unsafe fn vec_nabs(self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-1"),
        assert_instr(vflnsb)
    )]
    unsafe fn vec_nabs_f32(a: vector_float) -> vector_float {
        simd_neg(simd_fabs(a))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vflndb))]
    unsafe fn vec_nabs_f64(a: vector_double) -> vector_double {
        simd_neg(simd_fabs(a))
    }

    impl_vec_trait! { [VectorNabs vec_nabs] vec_nabs_f32 (vector_float) }
    impl_vec_trait! { [VectorNabs vec_nabs] vec_nabs_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSplat {
        unsafe fn vec_splat<const IMM: u32>(self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepb, IMM2 = 1))]
    unsafe fn vrepb<const IMM2: u32>(a: vector_signed_char) -> vector_signed_char {
        static_assert_uimm_bits!(IMM2, 4);
        simd_shuffle(a, a, const { u32x16::from_array([IMM2; 16]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vreph, IMM2 = 1))]
    unsafe fn vreph<const IMM2: u32>(a: vector_signed_short) -> vector_signed_short {
        static_assert_uimm_bits!(IMM2, 3);
        simd_shuffle(a, a, const { u32x8::from_array([IMM2; 8]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepf, IMM2 = 1))]
    unsafe fn vrepf<const IMM2: u32>(a: vector_signed_int) -> vector_signed_int {
        static_assert_uimm_bits!(IMM2, 2);
        simd_shuffle(a, a, const { u32x4::from_array([IMM2; 4]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepg, IMM2 = 1))]
    unsafe fn vrepg<const IMM2: u32>(a: vector_signed_long_long) -> vector_signed_long_long {
        static_assert_uimm_bits!(IMM2, 1);
        simd_shuffle(a, a, const { u32x2::from_array([IMM2; 2]) })
    }

    macro_rules! impl_vec_splat {
        ($ty:ty, $fun:ident) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorSplat for $ty {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_splat<const IMM: u32>(self) -> Self {
                    transmute($fun::<IMM>(transmute(self)))
                }
            }
        };
    }

    impl_vec_splat! { vector_signed_char, vrepb }
    impl_vec_splat! { vector_unsigned_char, vrepb }
    impl_vec_splat! { vector_bool_char, vrepb }
    impl_vec_splat! { vector_signed_short, vreph }
    impl_vec_splat! { vector_unsigned_short, vreph }
    impl_vec_splat! { vector_bool_short, vreph }
    impl_vec_splat! { vector_signed_int, vrepf }
    impl_vec_splat! { vector_unsigned_int, vrepf }
    impl_vec_splat! { vector_bool_int, vrepf }
    impl_vec_splat! { vector_signed_long_long, vrepg }
    impl_vec_splat! { vector_unsigned_long_long, vrepg }
    impl_vec_splat! { vector_bool_long_long, vrepg }

    impl_vec_splat! { vector_float, vrepf }
    impl_vec_splat! { vector_double, vrepg }

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

    test_impl! { vec_vpopctb_signed +(a: vector_signed_char) -> vector_signed_char [simd_ctpop, vpopctb] }
    test_impl! { vec_vpopcth_signed +(a: vector_signed_short) -> vector_signed_short [simd_ctpop, "vector-enhancements-1" vpopcth] }
    test_impl! { vec_vpopctf_signed +(a: vector_signed_int) -> vector_signed_int [simd_ctpop, "vector-enhancements-1" vpopctf] }
    test_impl! { vec_vpopctg_signed +(a: vector_signed_long_long) -> vector_signed_long_long [simd_ctpop, "vector-enhancements-1" vpopctg] }

    test_impl! { vec_vpopctb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_ctpop, vpopctb] }
    test_impl! { vec_vpopcth_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_ctpop, "vector-enhancements-1" vpopcth] }
    test_impl! { vec_vpopctf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_ctpop, "vector-enhancements-1" vpopctf] }
    test_impl! { vec_vpopctg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_ctpop, "vector-enhancements-1" vpopctg] }

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
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vno))]
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
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnn))]
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
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnx))]
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
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnc))]
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
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(voc))]
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

    test_impl! { vec_roundc_f32 (a: vector_float) -> vector_float [nearbyint_v4f32,  "vector-enhancements-1" vfisb] }
    test_impl! { vec_roundc_f64 (a: vector_double) -> vector_double [nearbyint_v2f64, vfidb] }

    // FIXME(llvm) roundeven does not yet lower to vfidb (but should in the future)
    test_impl! { vec_round_f32 (a: vector_float) -> vector_float [roundeven_v4f32, _] }
    test_impl! { vec_round_f64 (a: vector_double) -> vector_double [roundeven_v2f64, _] }

    test_impl! { vec_rint_f32 (a: vector_float) -> vector_float [rint_v4f32, "vector-enhancements-1" vfisb] }
    test_impl! { vec_rint_f64 (a: vector_double) -> vector_double [rint_v2f64, vfidb] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRoundc {
        unsafe fn vec_roundc(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRound {
        unsafe fn vec_round(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRint {
        unsafe fn vec_rint(self) -> Self;
    }

    impl_vec_trait! { [VectorRoundc vec_roundc] vec_roundc_f32 (vector_float) }
    impl_vec_trait! { [VectorRoundc vec_roundc] vec_roundc_f64 (vector_double) }

    impl_vec_trait! { [VectorRound vec_round] vec_round_f32 (vector_float) }
    impl_vec_trait! { [VectorRound vec_round] vec_round_f64 (vector_double) }

    impl_vec_trait! { [VectorRint vec_rint] vec_rint_f32 (vector_float) }
    impl_vec_trait! { [VectorRint vec_rint] vec_rint_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorTrunc {
        // same as vec_roundz
        unsafe fn vec_trunc(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCeil {
        // same as vec_roundp
        unsafe fn vec_ceil(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFloor {
        // same as vec_roundm
        unsafe fn vec_floor(self) -> Self;
    }

    impl_vec_trait! { [VectorTrunc vec_trunc] simd_trunc (vector_float) }
    impl_vec_trait! { [VectorTrunc vec_trunc] simd_trunc (vector_double) }

    impl_vec_trait! { [VectorCeil vec_ceil] simd_ceil (vector_float) }
    impl_vec_trait! { [VectorCeil vec_ceil] simd_ceil (vector_double) }

    impl_vec_trait! { [VectorFloor vec_floor] simd_floor (vector_float) }
    impl_vec_trait! { [VectorFloor vec_floor] simd_floor (vector_double) }

    macro_rules! impl_vec_shift {
        ([$Trait:ident $m:ident] ($b:ident, $h:ident, $w:ident, $g:ident)) => {
            impl_vec_trait!{ [$Trait $m]+ $b (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$Trait $m]+ $b (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$Trait $m]+ $h (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$Trait $m]+ $h (vector_signed_short, vector_unsigned_short) -> vector_signed_short }
            impl_vec_trait!{ [$Trait $m]+ $w (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$Trait $m]+ $w (vector_signed_int, vector_unsigned_int) -> vector_signed_int }
            impl_vec_trait!{ [$Trait $m]+ $g (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$Trait $m]+ $g (vector_signed_long_long, vector_unsigned_long_long) -> vector_signed_long_long }
        };
    }

    macro_rules! impl_shift {
        ($fun:ident $intr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($fun))]
            unsafe fn $fun(a: t_t_l!($ty), b: t_t_l!($ty)) -> t_t_l!($ty) {
                let a = transmute(a);
                // use the remainder of b by the width of a's elements to prevent UB
                let b = simd_rem(transmute(b), <t_t_s!($ty)>::splat($ty::BITS as $ty));

                transmute($intr(a, b))
            }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSl<Other> {
        type Result;
        unsafe fn vec_sl(self, b: Other) -> Self::Result;
    }

    impl_shift! { veslvb simd_shl u8 }
    impl_shift! { veslvh simd_shl u16 }
    impl_shift! { veslvf simd_shl u32 }
    impl_shift! { veslvg simd_shl u64 }

    impl_vec_shift! { [VectorSl vec_sl] (veslvb, veslvh, veslvf, veslvg) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSr<Other> {
        type Result;
        unsafe fn vec_sr(self, b: Other) -> Self::Result;
    }

    impl_shift! { vesrlvb simd_shr u8 }
    impl_shift! { vesrlvh simd_shr u16 }
    impl_shift! { vesrlvf simd_shr u32 }
    impl_shift! { vesrlvg simd_shr u64 }

    impl_vec_shift! { [VectorSr vec_sr] (vesrlvb, vesrlvh, vesrlvf, vesrlvg) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSra<Other> {
        type Result;
        unsafe fn vec_sra(self, b: Other) -> Self::Result;
    }

    impl_shift! { vesravb simd_shr i8 }
    impl_shift! { vesravh simd_shr i16 }
    impl_shift! { vesravf simd_shr i32 }
    impl_shift! { vesravg simd_shr i64 }

    impl_vec_shift! { [VectorSra vec_sra] (vesravb, vesravh, vesravf, vesravg) }

    macro_rules! impl_vec_shift_byte {
        ([$trait:ident $m:ident] ($f:ident)) => {
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_signed_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_signed_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_signed_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_signed_short) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_unsigned_short) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_signed_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_signed_int) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_unsigned_int) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_signed_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_signed_long_long) -> vector_signed_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_unsigned_long_long) -> vector_signed_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_float, vector_signed_int) -> vector_float }
            impl_vec_trait!{ [$trait $m]+ $f (vector_float, vector_unsigned_int) -> vector_float }
            impl_vec_trait!{ [$trait $m]+ $f (vector_double, vector_signed_long_long) -> vector_double }
            impl_vec_trait!{ [$trait $m]+ $f (vector_double, vector_unsigned_long_long) -> vector_double }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSlb<Other> {
        type Result;
        unsafe fn vec_slb(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSlb vec_slb] (vslb) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrab<Other> {
        type Result;
        unsafe fn vec_srab(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSrab vec_srab] (vsrab) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrb<Other> {
        type Result;
        unsafe fn vec_srb(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSrb vec_srb] (vsrlb) }

    macro_rules! impl_vec_shift_long {
        ([$trait:ident $m:ident] ($f:ident)) => {
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_unsigned_char) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_unsigned_char) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_unsigned_char) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_unsigned_char) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_unsigned_char) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_unsigned_char) -> vector_signed_long_long }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrl<Other> {
        type Result;
        unsafe fn vec_srl(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSrl vec_srl] (vsrl) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSral<Other> {
        type Result;
        unsafe fn vec_sral(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSral vec_sral] (vsra) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSll<Other> {
        type Result;
        unsafe fn vec_sll(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSll vec_sll] (vsl) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRl<Other> {
        type Result;
        unsafe fn vec_rl(self, b: Other) -> Self::Result;
    }

    macro_rules! impl_rot {
        ($fun:ident $intr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($fun))]
            unsafe fn $fun(a: t_t_l!($ty), b: t_t_l!($ty)) -> t_t_l!($ty) {
                transmute($intr(transmute(a), transmute(a), transmute(b)))
            }
        };
    }

    impl_rot! { verllvb fshlb u8 }
    impl_rot! { verllvh fshlh u16 }
    impl_rot! { verllvf fshlf u32 }
    impl_rot! { verllvg fshlg u64 }

    impl_vec_shift! { [VectorRl vec_rl] (verllvb, verllvh, verllvf, verllvg) }

    macro_rules! test_rot_imm {
        ($fun:ident $instr:ident $intr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn $fun(a: t_t_l!($ty), bits: core::ffi::c_ulong) -> t_t_l!($ty) {
                // mod by the number of bits in a's element type to prevent UB
                let bits = (bits % $ty::BITS as core::ffi::c_ulong) as $ty;
                let a = transmute(a);
                let b = <t_t_s!($ty)>::splat(bits);

                transmute($intr(a, a, transmute(b)))
            }
        };
    }

    test_rot_imm! { verllvb_imm verllb fshlb u8 }
    test_rot_imm! { verllvh_imm verllh fshlh u16 }
    test_rot_imm! { verllvf_imm verllf fshlf u32 }
    test_rot_imm! { verllvg_imm verllg fshlg u64 }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRli {
        unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self;
    }

    macro_rules! impl_rot_imm {
        ($($ty:ident, $intr:ident),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRli for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self {
                        transmute($intr(transmute(self), bits))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRli for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self {
                        $intr(self, bits)
                    }
                }
            )*
        }
    }

    impl_rot_imm! {
        vector_signed_char, verllvb_imm,
        vector_signed_short, verllvh_imm,
        vector_signed_int, verllvf_imm,
        vector_signed_long_long, verllvg_imm
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRlMask<Other> {
        unsafe fn vec_rl_mask<const IMM8: u8>(self, other: Other) -> Self;
    }

    macro_rules! impl_rl_mask {
        ($($ty:ident, $intr:ident, $fun:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($intr, IMM8 = 6))]
                unsafe fn $fun<const IMM8: u8>(a: $ty, b: t_u!($ty)) -> $ty {
                    // mod by the number of bits in a's element type to prevent UB
                    $intr(a, a, transmute(b), const { (IMM8 % <l_t_t!($ty)>::BITS as u8) as i32 })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRlMask<t_u!($ty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rl_mask<const IMM8: u8>(self, other: t_u!($ty)) -> Self {
                        $fun::<IMM8>(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRlMask<t_u!($ty)> for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rl_mask<const IMM8: u8>(self, other: t_u!($ty)) -> Self {
                        transmute($fun::<IMM8>(transmute(self), transmute(other)))
                    }
                }
            )*
        }
    }

    impl_rl_mask! {
        vector_signed_char, verimb, test_verimb,
        vector_signed_short, verimh, test_verimh,
        vector_signed_int, verimf, test_verimf,
        vector_signed_long_long, verimg, test_verimg
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorReve {
        unsafe fn vec_reve(self) -> Self;
    }

    macro_rules! impl_reve {
        ($($ty:ident, $fun:ident, $instr:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($instr))]
                unsafe fn $fun(a: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, a, const { ShuffleMask::<N>::reverse() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        $fun(self)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        transmute($fun(transmute(self)))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for t_b!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        transmute($fun(transmute(self)))
                    }
                }
            )*
        }
    }

    impl_reve! {
        vector_signed_char, reveb, vperm,
        vector_signed_short, reveh, vperm,
        vector_signed_int, revef, vperm,
        vector_signed_long_long, reveg, vpdi
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorReve for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_reve(self) -> Self {
            transmute(transmute::<_, vector_signed_int>(self).vec_reve())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorReve for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_reve(self) -> Self {
            transmute(transmute::<_, vector_signed_long_long>(self).vec_reve())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRevb {
        unsafe fn vec_revb(self) -> Self;
    }

    test_impl! { bswapb (a: vector_signed_char) -> vector_signed_char [simd_bswap, _] }
    test_impl! { bswaph (a: vector_signed_short) -> vector_signed_short [simd_bswap, vperm] }
    test_impl! { bswapf (a: vector_signed_int) -> vector_signed_int [simd_bswap, vperm] }
    test_impl! { bswapg (a: vector_signed_long_long) -> vector_signed_long_long [simd_bswap, vperm] }

    impl_vec_trait! { [VectorRevb vec_revb]+ bswapb (vector_unsigned_char) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapb (vector_signed_char) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswaph (vector_unsigned_short) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswaph (vector_signed_short) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapf (vector_unsigned_int) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapf (vector_signed_int) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapg (vector_unsigned_long_long) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapg (vector_signed_long_long) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRevb for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_revb(self) -> Self {
            transmute(transmute::<_, vector_signed_int>(self).vec_revb())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRevb for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_revb(self) -> Self {
            transmute(transmute::<_, vector_signed_long_long>(self).vec_revb())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMergel {
        unsafe fn vec_mergel(self, other: Self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMergeh {
        unsafe fn vec_mergeh(self, other: Self) -> Self;
    }

    macro_rules! impl_merge {
        ($($ty:ident, $mergel:ident, $mergeh:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($mergel))]
                unsafe fn $mergel(a: $ty, b: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, b, const { ShuffleMask::<N>::merge_low() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergel for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergel(self, other: Self) -> Self {
                        $mergel(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergel for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergel(self, other: Self) -> Self {
                        transmute($mergel(transmute(self), transmute(other)))
                    }
                }

                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($mergeh))]
                unsafe fn $mergeh(a: $ty, b: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, b, const { ShuffleMask::<N>::merge_high() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergeh for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergeh(self, other: Self) -> Self {
                        $mergeh(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergeh for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergeh(self, other: Self) -> Self {
                        transmute($mergeh(transmute(self), transmute(other)))
                    }
                }
            )*
        }
    }

    impl_merge! {
        vector_signed_char, vmrlb, vmrhb,
        vector_signed_short, vmrlh, vmrhh,
        vector_signed_int, vmrlf, vmrhf,
        vector_signed_long_long, vmrlg, vmrhg
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorPerm {
        unsafe fn vec_perm(self, other: Self, c: vector_unsigned_char) -> Self;
    }

    macro_rules! impl_merge {
        ($($ty:ident),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorPerm for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_perm(self, other: Self, c: vector_unsigned_char) -> Self {
                        transmute(vperm(transmute(self), transmute(other), c))
                    }
                }
            )*
        }
    }

    impl_merge! {
        vector_signed_char,
        vector_signed_short,
        vector_signed_int,
        vector_signed_long_long,
        vector_unsigned_char,
        vector_unsigned_short,
        vector_unsigned_int,
        vector_unsigned_long_long,
        vector_bool_char,
        vector_bool_short,
        vector_bool_int,
        vector_bool_long_long,
        vector_float,
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSumU128 {
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsumqf))]
    pub unsafe fn vec_vsumqf(a: vector_unsigned_int, b: vector_unsigned_int) -> u128 {
        transmute(vsumqf(a, b))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsumqg))]
    pub unsafe fn vec_vsumqg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> u128 {
        transmute(vsumqg(a, b))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSumU128 for vector_unsigned_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char {
            transmute(vec_vsumqf(self, other))
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSumU128 for vector_unsigned_long_long {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char {
            transmute(vec_vsumqg(self, other))
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSum2 {
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long;
    }

    test_impl! { vec_vsumgh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_long_long [vsumgh, vsumgh] }
    test_impl! { vec_vsumgf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long [vsumgf, vsumgf] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum2 for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long {
            vec_vsumgh(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum2 for vector_unsigned_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long {
            vec_vsumgf(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSum4 {
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int;
    }

    test_impl! { vec_vsumb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_int [vsumb, vsumb] }
    test_impl! { vec_vsumh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int [vsumh, vsumh] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum4 for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int {
            vec_vsumb(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum4 for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int {
            vec_vsumh(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSubc<Other> {
        type Result;
        unsafe fn vec_subc(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vscbib (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vscbib, vscbib] }
    test_impl! { vec_vscbih (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vscbih, vscbih] }
    test_impl! { vec_vscbif (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vscbif, vscbif] }
    test_impl! { vec_vscbig (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vscbig, vscbig] }

    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbib (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbih (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbif (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbig (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSqrt {
        unsafe fn vec_sqrt(self) -> Self;
    }

    test_impl! { vec_sqrt_f32 (v: vector_float) -> vector_float [ simd_fsqrt, "vector-enhancements-1" vfsqsb ] }
    test_impl! { vec_sqrt_f64 (v: vector_double) -> vector_double [ simd_fsqrt, vfsqdb ] }

    impl_vec_trait! { [VectorSqrt vec_sqrt] vec_sqrt_f32 (vector_float) }
    impl_vec_trait! { [VectorSqrt vec_sqrt] vec_sqrt_f64 (vector_double) }

    macro_rules! vfae_wrapper {
        ($($name:ident $ty:ident)*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($name, IMM = 0))]
                unsafe fn $name<const IMM: i32>(
                    a: $ty,
                    b: $ty,
                ) -> $ty {
                    super::$name(a, b, IMM)
                }
            )*
        }
     }

    vfae_wrapper! {
       vfaeb vector_signed_char
       vfaeh vector_signed_short
       vfaef vector_signed_int

       vfaezb vector_signed_char
       vfaezh vector_signed_short
       vfaezf vector_signed_int
    }

    macro_rules! impl_vfae {
        ([idx_cc $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [idx_cc $Trait $m] $imm
                $b vector_signed_char vector_signed_char
                $b vector_unsigned_char vector_unsigned_char
                $b vector_bool_char vector_unsigned_char

                $h vector_signed_short vector_signed_short
                $h vector_unsigned_short vector_unsigned_short
                $h vector_bool_short vector_unsigned_short

                $f vector_signed_int vector_signed_int
                $f vector_unsigned_int vector_unsigned_int
                $f vector_bool_int vector_unsigned_int
            }
        };
        ([idx_cc $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident $r:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = $r;
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self, c: *mut i32) -> Self::Result {
                        let PackedTuple { x, y } = $fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b));
                        c.write(y);
                        transmute(x)
                    }
                }
            )*
        };
        ([cc $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [cc $Trait $m] $imm
                $b vector_signed_char
                $b vector_unsigned_char
                $b vector_bool_char

                $h vector_signed_short
                $h vector_unsigned_short
                $h vector_bool_short

                $f vector_signed_int
                $f vector_unsigned_int
                $f vector_bool_int
            }
        };
        ([cc $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = t_b!($ty);
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self, c: *mut i32) -> Self::Result {
                        let PackedTuple { x, y } = $fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b));
                        c.write(y);
                        transmute(x)
                    }
                }
            )*
        };
        ([idx $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [idx $Trait $m] $imm
                $b vector_signed_char vector_signed_char
                $b vector_unsigned_char vector_unsigned_char
                $b vector_bool_char vector_unsigned_char

                $h vector_signed_short vector_signed_short
                $h vector_unsigned_short vector_unsigned_short
                $h vector_bool_short vector_unsigned_short

                $f vector_signed_int vector_signed_int
                $f vector_unsigned_int vector_unsigned_int
                $f vector_bool_int vector_unsigned_int
            }
        };
        ([idx $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident $r:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = $r;
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> Self::Result {
                        transmute($fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b)))
                    }
                }
            )*
        };
        ([$Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [$Trait $m] $imm
                $b vector_signed_char
                $b vector_unsigned_char
                $b vector_bool_char

                $h vector_signed_short
                $h vector_unsigned_short
                $h vector_bool_short

                $f vector_signed_int
                $f vector_unsigned_int
                $f vector_bool_int
            }
        };
        ([$Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = t_b!($ty);
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> Self::Result {
                        transmute($fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b)))
                    }
                }
            )*
        };
    }

    enum FindImm {
        Eq = 4,
        Ne = 12,
        EqIdx = 0,
        NeIdx = 8,
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEq<Other> {
        type Result;
        unsafe fn vec_find_any_eq(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [VectorFindAnyEq vec_find_any_eq] Eq vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNe<Other> {
        type Result;
        unsafe fn vec_find_any_ne(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [VectorFindAnyNe vec_find_any_ne] Ne vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqOrZeroIdx<Other> {
        type Result;
        unsafe fn vec_find_any_eq_or_0_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyEqOrZeroIdx vec_find_any_eq_or_0_idx] EqIdx
        vfaezb vector_signed_char vector_signed_char
        vfaezb vector_unsigned_char vector_unsigned_char
        vfaezb vector_bool_char vector_unsigned_char

        vfaezh vector_signed_short vector_signed_short
        vfaezh vector_unsigned_short vector_unsigned_short
        vfaezh vector_bool_short vector_unsigned_short

        vfaezf vector_signed_int vector_signed_int
        vfaezf vector_unsigned_int vector_unsigned_int
        vfaezf vector_bool_int vector_unsigned_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeOrZeroIdx<Other> {
        type Result;
        unsafe fn vec_find_any_ne_or_0_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyNeOrZeroIdx vec_find_any_ne_or_0_idx] NeIdx
        vfaezb vector_signed_char vector_signed_char
        vfaezb vector_unsigned_char vector_unsigned_char
        vfaezb vector_bool_char vector_unsigned_char

        vfaezh vector_signed_short vector_signed_short
        vfaezh vector_unsigned_short vector_unsigned_short
        vfaezh vector_bool_short vector_unsigned_short

        vfaezf vector_signed_int vector_signed_int
        vfaezf vector_unsigned_int vector_unsigned_int
        vfaezf vector_bool_int vector_unsigned_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqIdx<Other> {
        type Result;
        unsafe fn vec_find_any_eq_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyEqIdx vec_find_any_eq_idx] EqIdx vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeIdx<Other> {
        type Result;
        unsafe fn vec_find_any_ne_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyNeIdx vec_find_any_ne_idx] NeIdx vfaeb vfaeh vfaef }

    macro_rules! vfaes_wrapper {
        ($($name:ident $ty:ident)*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($name, IMM = 0))]
                unsafe fn $name<const IMM: i32>(
                    a: $ty,
                    b: $ty,
                ) -> PackedTuple<$ty, i32> {
                    super::$name(a, b, IMM)
                }
            )*
        }
     }

    vfaes_wrapper! {
       vfaebs vector_signed_char
       vfaehs vector_signed_short
       vfaefs vector_signed_int

       vfaezbs vector_signed_char
       vfaezhs vector_signed_short
       vfaezfs vector_signed_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [cc VectorFindAnyEqCC vec_find_any_eq_cc] Eq vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [cc VectorFindAnyNeCC vec_find_any_ne_cc] Ne vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_idx_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [idx_cc VectorFindAnyEqIdxCC vec_find_any_eq_idx_cc] EqIdx vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_idx_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [idx_cc VectorFindAnyNeIdxCC vec_find_any_ne_idx_cc] NeIdx vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqOrZeroIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_or_0_idx_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [idx_cc VectorFindAnyEqOrZeroIdxCC vec_find_any_eq_or_0_idx_cc] EqIdx vfaezbs vfaezhs vfaezfs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeOrZeroIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_or_0_idx_cc(self, other: Other, c: *mut i32) -> Self::Result;
    }

    impl_vfae! { [idx_cc VectorFindAnyNeOrZeroIdxCC vec_find_any_ne_or_0_idx_cc] NeIdx vfaezbs vfaezhs vfaezfs }
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

/// Vector negative abs.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nabs<T>(a: T) -> T
where
    T: sealed::VectorNabs,
{
    a.vec_nabs()
}

/// Vector square root.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sqrt<T>(a: T) -> T
where
    T: sealed::VectorSqrt,
{
    a.vec_sqrt()
}

/// Vector Splat
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_splat<T, const IMM: u32>(a: T) -> T
where
    T: sealed::VectorSplat,
{
    a.vec_splat::<IMM>()
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
    T: sealed::VectorFloor,
{
    a.vec_floor()
}

/// Vector ceil.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_ceil<T>(a: T) -> T
where
    T: sealed::VectorCeil,
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
    T: sealed::VectorTrunc,
{
    a.vec_trunc()
}

/// Returns a vector containing the rounded values to the nearest representable floating-point integer,
/// using IEEE round-to-nearest rounding, of the corresponding elements of the given vector
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
    T: sealed::VectorRoundc,
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
    T: sealed::VectorFloor,
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
    T: sealed::VectorCeil,
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
    T: sealed::VectorTrunc,
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
    T: sealed::VectorRint,
{
    a.vec_rint()
}

/// Vector Shift Left
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sl<T, U>(a: T, b: U) -> <T as sealed::VectorSl<U>>::Result
where
    T: sealed::VectorSl<U>,
{
    a.vec_sl(b)
}

/// Vector Shift Right
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sr<T, U>(a: T, b: U) -> <T as sealed::VectorSr<U>>::Result
where
    T: sealed::VectorSr<U>,
{
    a.vec_sr(b)
}

/// Vector Shift Right Algebraic
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sra<T, U>(a: T, b: U) -> <T as sealed::VectorSra<U>>::Result
where
    T: sealed::VectorSra<U>,
{
    a.vec_sra(b)
}

/// Vector Shift Left by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_slb<T, U>(a: T, b: U) -> <T as sealed::VectorSlb<U>>::Result
where
    T: sealed::VectorSlb<U>,
{
    a.vec_slb(b)
}

/// Vector Shift Right by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srb<T, U>(a: T, b: U) -> <T as sealed::VectorSrb<U>>::Result
where
    T: sealed::VectorSrb<U>,
{
    a.vec_srb(b)
}

/// Vector Shift Right Algebraic by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srab<T, U>(a: T, b: U) -> <T as sealed::VectorSrab<U>>::Result
where
    T: sealed::VectorSrab<U>,
{
    a.vec_srab(b)
}

/// Vector Element Rotate Left
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rl<T, U>(a: T, b: U) -> <T as sealed::VectorRl<U>>::Result
where
    T: sealed::VectorRl<U>,
{
    a.vec_rl(b)
}

/// Performs a left shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a left by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by zeros.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sll<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSll<vector_unsigned_char, Result = T>,
{
    a.vec_sll(b)
}

/// Performs a right shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a right by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by zeros.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srl<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSrl<vector_unsigned_char, Result = T>,
{
    a.vec_srl(b)
}

/// Performs an algebraic right shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a right by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by copies of
/// the most significant bit of the element of a.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sral<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSral<vector_unsigned_char, Result = T>,
{
    a.vec_sral(b)
}

/// Rotates each element of a vector left by a given number of bits. Each element of the result is obtained by rotating the corresponding element
/// of a left by the number of bits specified by b, modulo the number of bits in the element.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rli<T>(a: T, bits: core::ffi::c_ulong) -> T
where
    T: sealed::VectorRli,
{
    a.vec_rli(bits)
}

/// Returns a vector with the elements of the input vector in reversed order.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_reve<T>(a: T) -> T
where
    T: sealed::VectorReve,
{
    a.vec_reve()
}

/// Returns a vector where each vector element contains the corresponding byte-reversed vector element of the input vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_revb<T>(a: T) -> T
where
    T: sealed::VectorRevb,
{
    a.vec_revb()
}

/// Merges the most significant ("high") halves of two vectors.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mergeh<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergeh,
{
    a.vec_mergeh(b)
}

/// Merges the least significant ("low") halves of two vectors.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mergel<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergel,
{
    a.vec_mergel(b)
}

/// Generates byte masks for elements in the return vector. For each bit in a, if the bit is one, all bit positions
/// in the corresponding byte element of d are set to ones. Otherwise, if the bit is zero, the corresponding byte element is set to zero.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgbm, MASK = 0x00FF))]
pub unsafe fn vec_genmask<const MASK: u16>() -> vector_unsigned_char {
    vector_unsigned_char(const { genmask::<MASK>() })
}

/// Vector Generate Mask (Byte)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, L = 3, H = 5))]
pub unsafe fn vec_genmasks_8<const L: u8, const H: u8>() -> vector_unsigned_char {
    vector_unsigned_char(const { [genmasks(u8::BITS, L, H) as u8; 16] })
}

/// Vector Generate Mask (Halfword)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, L = 3, H = 5))]
pub unsafe fn vec_genmasks_16<const L: u8, const H: u8>() -> vector_unsigned_short {
    vector_unsigned_short(const { [genmasks(u16::BITS, L, H) as u16; 8] })
}

/// Vector Generate Mask (Word)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgmf, L = 3, H = 5))]
pub unsafe fn vec_genmasks_32<const L: u8, const H: u8>() -> vector_unsigned_int {
    vector_unsigned_int(const { [genmasks(u32::BITS, L, H) as u32; 4] })
}

/// Vector Generate Mask (Doubleword)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgmg, L = 3, H = 5))]
pub unsafe fn vec_genmasks_64<const L: u8, const H: u8>() -> vector_unsigned_long_long {
    vector_unsigned_long_long(const { [genmasks(u64::BITS, L, H); 2] })
}

/// Returns a vector that contains some elements of two vectors, in the order specified by a third vector.
/// Each byte of the result is selected by using the least significant 5 bits of the corresponding byte of c as an index into the concatenated bytes of a and b.
/// Note: The vector generate mask built-in function [`vec_genmask`] could help generate the mask c.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_perm<T: sealed::VectorPerm>(a: T, b: T, c: vector_unsigned_char) -> T {
    a.vec_perm(b, c)
}

/// Vector Sum Across Quadword
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the quadword of vector a,
/// and the rightmost word or doubleword element of the b. The result is an unsigned 128-bit integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum_u128<T: sealed::VectorSumU128>(a: T, b: T) -> vector_unsigned_char {
    a.vec_sum_u128(b)
}

/// Vector Sum Across Doubleword
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the doubleword of vector a,
/// and the rightmost sub-element of the corresponding doubleword of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum2<T: sealed::VectorSum2>(a: T, b: T) -> vector_unsigned_long_long {
    a.vec_sum2(b)
}

/// Vector Sum Across Word
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the word of vector a,
/// and the rightmost sub-element of the corresponding word of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum4<T: sealed::VectorSum4>(a: T, b: T) -> vector_unsigned_int {
    a.vec_sum4(b)
}

/// Vector Subtract unsigned 128-bits
///
/// Subtracts unsigned quadword values.
///
/// This function operates on the vectors as 128-bit unsigned integers. It returns low 128 bits of a - b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsq))]
pub unsafe fn vec_sub_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);

    transmute(a.wrapping_sub(b))
}

/// Vector Subtract Carryout
///
/// Returns a vector containing the borrow produced by subtracting each of corresponding elements of b from a.
///
/// On each resulting element, the value is 0 if a borrow occurred, or 1 if no borrow occurred.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_subc<T, U>(a: T, b: U) -> <T as sealed::VectorSubc<U>>::Result
where
    T: sealed::VectorSubc<U>,
{
    a.vec_subc(b)
}

/// Gets the carry bit of the 128-bit subtraction of two quadword values.
/// This function operates on the vectors as 128-bit unsigned integers. It returns a vector containing the borrow produced by subtracting b from a, as unsigned 128-bits integers.
/// If no borrow occurred, the bit 127 of d is 1; otherwise it is set to 0. All other bits of d are 0.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vscbiq))]
pub unsafe fn vec_subc_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vscbiq(transmute(a), transmute(b)))
}

/// Subtracts unsigned quadword values with carry bit from a previous operation.
///
/// This function operates on the vectors as 128-bit unsigned integers. It returns a vector containing the result of subtracting of b from a,
/// and the carryout bit from a previous operation.
///
/// Note: Only the borrow indication bit (127-bit) of c is used, and the other bits are ignored.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsbiq))]
pub unsafe fn vec_sube_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vsbiq(transmute(a), transmute(b), transmute(c)))
}

/// Vector Subtract with Carryout, Carryout
///
/// Gets the carry bit of the 128-bit subtraction of two quadword values with carry bit from the previous operation.
///
/// It returns a vector containing the carryout produced from the result of subtracting of b from a,
/// and the carryout bit from a previous operation. If no borrow occurred, the 127-bit of d is 1, otherwise 0.
/// All other bits of d are 0.
///
/// Note: Only the borrow indication bit (127-bit) of c is used, and the other bits are ignored.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsbcbiq))]
pub unsafe fn vec_subec_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vsbcbiq(transmute(a), transmute(b), transmute(c)))
}

/// Vector Splat Signed Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, IMM = 42))]
pub unsafe fn vec_splat_i8<const IMM: i8>() -> vector_signed_char {
    vector_signed_char([IMM; 16])
}

/// Vector Splat Signed Halfword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, IMM = 42))]
pub unsafe fn vec_splat_i16<const IMM: i16>() -> vector_signed_short {
    vector_signed_short([IMM as i16; 8])
}

/// Vector Splat Signed Word
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepif, IMM = 42))]
pub unsafe fn vec_splat_i32<const IMM: i16>() -> vector_signed_int {
    vector_signed_int([IMM as i32; 4])
}

/// Vector Splat Signed Doubleword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepig, IMM = 42))]
pub unsafe fn vec_splat_i64<const IMM: i16>() -> vector_signed_long_long {
    vector_signed_long_long([IMM as i64; 2])
}

/// Vector Splat Unsigned Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, IMM = 42))]
pub unsafe fn vec_splat_u8<const IMM: u8>() -> vector_unsigned_char {
    vector_unsigned_char([IMM; 16])
}

/// Vector Splat Unsigned Halfword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, IMM = 42))]
pub unsafe fn vec_splat_u16<const IMM: i16>() -> vector_unsigned_short {
    vector_unsigned_short([IMM as u16; 8])
}

/// Vector Splat Unsigned Word
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepif, IMM = 42))]
pub unsafe fn vec_splat_u32<const IMM: i16>() -> vector_unsigned_int {
    vector_unsigned_int([IMM as u32; 4])
}

/// Vector Splat Unsigned Doubleword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepig, IMM = 42))]
pub unsafe fn vec_splat_u64<const IMM: i16>() -> vector_unsigned_long_long {
    vector_unsigned_long_long([IMM as u64; 2])
}

macro_rules! vec_find_any {
    ($($Trait:ident $fun:ident)*) => {
        $(
            #[inline]
            #[target_feature(enable = "vector")]
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            pub unsafe fn $fun<T, U>(a: T, b: U) -> <T as sealed::$Trait<U>>::Result
            where
                T: sealed::$Trait<U>,
            {
                a.$fun(b)
            }
        )*
    }
}

vec_find_any! {
    VectorFindAnyEq vec_find_any_eq
    VectorFindAnyNe vec_find_any_ne
    VectorFindAnyEqIdx vec_find_any_eq_idx
    VectorFindAnyNeIdx vec_find_any_ne_idx
    VectorFindAnyEqOrZeroIdx vec_find_any_eq_or_0_idx
    VectorFindAnyNeOrZeroIdx vec_find_any_ne_or_0_idx
}

macro_rules! vec_find_any_cc {
    ($($Trait:ident $fun:ident)*) => {
        $(
            #[inline]
            #[target_feature(enable = "vector")]
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            pub unsafe fn $fun<T, U>(a: T, b: U, c: *mut i32) -> <T as sealed::$Trait<U>>::Result
            where
                T: sealed::$Trait<U>,
            {
                a.$fun(b, c)
            }
        )*
    }
}

vec_find_any_cc! {
    VectorFindAnyEqCC vec_find_any_eq_cc
    VectorFindAnyNeCC vec_find_any_ne_cc
    VectorFindAnyEqIdxCC vec_find_any_eq_idx_cc
    VectorFindAnyNeIdxCC vec_find_any_ne_idx_cc
    VectorFindAnyEqOrZeroIdxCC vec_find_any_eq_or_0_idx_cc
    VectorFindAnyNeOrZeroIdxCC vec_find_any_ne_or_0_idx_cc
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    #[test]
    fn reverse_mask() {
        assert_eq!(ShuffleMask::<4>::reverse().0, [3, 2, 1, 0]);
    }

    #[test]
    fn mergel_mask() {
        assert_eq!(ShuffleMask::<4>::merge_low().0, [2, 6, 3, 7]);
    }

    #[test]
    fn mergeh_mask() {
        assert_eq!(ShuffleMask::<4>::merge_high().0, [0, 4, 1, 5]);
    }

    #[test]
    fn test_vec_mask() {
        assert_eq!(
            genmask::<0x00FF>(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
            ]
        );
    }

    #[test]
    fn test_genmasks() {
        assert_eq!(genmasks(u8::BITS, 3, 5), 28);
        assert_eq!(genmasks(u8::BITS, 3, 7), 31);

        // If a or b is greater than 8, the operation is performed as if the value gets modulo by 8.
        assert_eq!(genmasks(u8::BITS, 3 + 8, 7 + 8), 31);
        // If a is greater than b, the operation is perform as if b equals 7.
        assert_eq!(genmasks(u8::BITS, 5, 4), genmasks(u8::BITS, 5, 7));

        assert_eq!(
            genmasks(u16::BITS, 4, 12) as u16,
            u16::from_be_bytes([15, -8i8 as u8])
        );
        assert_eq!(
            genmasks(u32::BITS, 4, 29) as u32,
            u32::from_be_bytes([15, 0xFF, 0xFF, -4i8 as u8])
        );
    }

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
            test_vec_2! { $name, $fn, $ty, $ty -> $ty, [$($a),+], [$($b),+], [$($d),+] }
         };
        { $name: ident, $fn:ident, $ty1: ident, $ty2: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty1) = transmute($ty1::new($($a),+));
                let b: s_t_l!($ty2) = transmute($ty2::new($($b),+));

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

    test_vec_1! { test_vec_nabs, vec_nabs, f32x4,
    [core::f32::consts::PI, 1.0, 0.0, -1.0],
    [-core::f32::consts::PI, -1.0, 0.0, -1.0] }

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

    test_vec_1! { test_vec_round_f32, vec_round, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_round_f32_even_odd, vec_round, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_round_f64_1, vec_round, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_round_f64_2, vec_round, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_1! { test_vec_roundc_f32, vec_roundc, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_roundc_f32_even_odd, vec_roundc, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_roundc_f64_1, vec_roundc, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_roundc_f64_2, vec_roundc, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_1! { test_vec_rint_f32, vec_rint, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_rint_f32_even_odd, vec_rint, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_rint_f64_1, vec_rint, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_rint_f64_2, vec_rint, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_2! { test_vec_sll, vec_sll, i32x4, u8x16 -> i32x4,
    [1, 1, 1, 1],
    [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 8],
    [1 << 2, 1 << 3, 1 << 4, 1] }

    test_vec_2! { test_vec_srl, vec_srl, i32x4, u8x16 -> i32x4,
    [0b1000, 0b1000, 0b1000, 0b1000],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [4, 2, 1, 8] }

    test_vec_2! { test_vec_sral_pos, vec_sral, u32x4, u8x16 -> i32x4,
    [0b1000, 0b1000, 0b1000, 0b1000],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [4, 2, 1, 8] }

    test_vec_2! { test_vec_sral_neg, vec_sral, i32x4, u8x16 -> i32x4,
    [-8, -8, -8, -8],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [-4, -2, -1, -8] }

    test_vec_2! { test_vec_rl, vec_rl, u32x4,
    [0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0x12345678],
    [4, 8, 12, 68],
    [0x23456781, 0xBCDEF09A, 0xF0F0F0F0, 0x23456781] }

    test_vec_1! { test_vec_reve_f32, vec_reve, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.9, 0.6, 0.5, 0.1]
    }

    test_vec_1! { test_vec_revb_u32, vec_revb, u32x4,
        [0xAABBCCDD, 0xEEFF0011, 0x22334455, 0x66778899],
        [0xDDCCBBAA, 0x1100FFEE, 0x55443322, 0x99887766]
    }

    test_vec_2! { test_vec_mergeh_u32, vec_mergeh, u32x4,
        [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD],
        [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        [0xAAAAAAAA, 0x00000000, 0xBBBBBBBB, 0x11111111]
    }

    test_vec_2! { test_vec_mergel_u32, vec_mergel, u32x4,
        [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD],
        [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        [0xCCCCCCCC, 0x22222222, 0xDDDDDDDD, 0x33333333]
    }

    macro_rules! test_vec_perm {
        {$name:ident,
         $shorttype:ident, $longtype:ident,
         [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: $longtype = transmute($shorttype::new($($a),+));
                let b: $longtype = transmute($shorttype::new($($b),+));
                let c: vector_unsigned_char = transmute(u8x16::new($($c),+));
                let d = $shorttype::new($($d),+);

                let r: $shorttype = transmute(vec_perm(a, b, c));
                assert_eq!(d, r);
            }
        }
    }

    test_vec_perm! {test_vec_perm_u8x16,
    u8x16, vector_unsigned_char,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105, 6, 7, 106, 107]}
    test_vec_perm! {test_vec_perm_i8x16,
    i8x16, vector_signed_char,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105, 6, 7, 106, 107]}

    test_vec_perm! {test_vec_perm_m8x16,
    m8x16, vector_bool_char,
    [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false],
    [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [false, false, true, true, false, false, true, true, false, false, true, true, false, false, true, true]}
    test_vec_perm! {test_vec_perm_u16x8,
    u16x8, vector_unsigned_short,
    [0, 1, 2, 3, 4, 5, 6, 7],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 10, 1, 11, 2, 12, 3, 13]}
    test_vec_perm! {test_vec_perm_i16x8,
    i16x8, vector_signed_short,
    [0, 1, 2, 3, 4, 5, 6, 7],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 10, 1, 11, 2, 12, 3, 13]}
    test_vec_perm! {test_vec_perm_m16x8,
    m16x8, vector_bool_short,
    [false, false, false, false, false, false, false, false],
    [true, true, true, true, true, true, true, true],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [false, true, false, true, false, true, false, true]}

    test_vec_perm! {test_vec_perm_u32x4,
    u32x4, vector_unsigned_int,
    [0, 1, 2, 3],
    [10, 11, 12, 13],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0, 10, 1, 11]}
    test_vec_perm! {test_vec_perm_i32x4,
    i32x4, vector_signed_int,
    [0, 1, 2, 3],
    [10, 11, 12, 13],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0, 10, 1, 11]}
    test_vec_perm! {test_vec_perm_m32x4,
    m32x4, vector_bool_int,
    [false, false, false, false],
    [true, true, true, true],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [false, true, false, true]}
    test_vec_perm! {test_vec_perm_f32x4,
    f32x4, vector_float,
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 1.1, 1.2, 1.3],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0.0, 1.0, 1.0, 1.1]}

    test_vec_1! { test_vec_sqrt, vec_sqrt, f32x4,
    [core::f32::consts::PI, 1.0, 25.0, 2.0],
    [core::f32::consts::PI.sqrt(), 1.0, 5.0, core::f32::consts::SQRT_2] }

    test_vec_2! { test_vec_find_any_eq, vec_find_any_eq, i32x4, i32x4 -> u32x4,
        [1, -2, 3, -4],
        [-5, 3, -7, 8],
        [0, 0, 0xFFFFFFFF, 0]
    }

    test_vec_2! { test_vec_find_any_ne, vec_find_any_ne, i32x4, i32x4 -> u32x4,
        [1, -2, 3, -4],
        [-5, 3, -7, 8],
        [0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF]
    }

    test_vec_2! { test_vec_find_any_eq_idx_1, vec_find_any_eq_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [5, 3, 7, 8],
        [0, 8, 0, 0]
    }
    test_vec_2! { test_vec_find_any_eq_idx_2, vec_find_any_eq_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 16, 0, 0]
    }

    test_vec_2! { test_vec_find_any_ne_idx_1, vec_find_any_ne_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [1, 5, 3, 4],
        [0, 4, 0, 0]
    }
    test_vec_2! { test_vec_find_any_ne_idx_2, vec_find_any_ne_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [0, 16, 0, 0]
    }

    test_vec_2! { test_vec_find_any_eq_or_0_idx_1, vec_find_any_eq_or_0_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 0, 4],
        [5, 6, 7, 8],
        [0, 8, 0, 0]
    }
    test_vec_2! { test_vec_find_any_ne_or_0_idx_1, vec_find_any_ne_or_0_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 0, 4],
        [1, 2, 3, 4],
        [0, 8, 0, 0]
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_cc() {
        let mut c = 0i32;

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let d = unsafe { vec_find_any_eq_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 0, -1, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let d = unsafe { vec_find_any_eq_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_cc() {
        let mut c = 0i32;

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let d = unsafe { vec_find_any_ne_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[-1, -1, 0, -1]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let d = unsafe { vec_find_any_ne_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_idx_cc() {
        let mut c = 0i32;

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let d = unsafe { vec_find_any_eq_idx_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 8, 0, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let d = unsafe { vec_find_any_eq_idx_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_idx_cc() {
        let mut c = 0i32;

        let a = vector_unsigned_int([5, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let d = unsafe { vec_find_any_ne_idx_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let d = unsafe { vec_find_any_ne_idx_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_or_0_idx_cc() {
        let mut c = 0i32;

        // if no element of a matches any element of b with an equal value, and there is at least one element from a with a value of 0
        let a = vector_unsigned_int([0, 1, 2, 3]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let d = unsafe { vec_find_any_eq_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 0);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if at least one element of a matches any element of b with an equal value, and no elements of a with a value of 0
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 2, 3, 4]);
        let d = unsafe { vec_find_any_eq_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        // if at least one element of a matches any element of b with an equal value, and there is at least one element from a has a value of 0
        let a = vector_unsigned_int([1, 2, 3, 0]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let d = unsafe { vec_find_any_eq_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 2);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if no element of a matches any element of b with an equal value, and there is no element from a with a value of 0.
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let d = unsafe { vec_find_any_eq_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_or_0_idx_cc() {
        let mut c = 0i32;

        // if no element of a matches any element of b with a not equal value, and there is at least one element from a with a value of 0.
        let a = vector_unsigned_int([0, 1, 2, 3]);
        let b = vector_unsigned_int([4, 1, 2, 3]);
        let d = unsafe { vec_find_any_ne_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 0);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if at least one element of a matches any element of b with a not equal value, and no elements of a with a value of 0.
        let a = vector_unsigned_int([4, 2, 3, 4]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let d = unsafe { vec_find_any_ne_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        // if at least one element of a matches any element of b with a not equal value, and there is at least one element from a has a value of 0.
        let a = vector_unsigned_int([1, 0, 1, 1]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let d = unsafe { vec_find_any_ne_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 2);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if no element of a matches any element of b with a not equal value, and there is no element from a with a value of 0.
        let a = vector_unsigned_int([4, 4, 4, 4]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let d = unsafe { vec_find_any_ne_or_0_idx_cc(a, b, &mut c) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }
}
