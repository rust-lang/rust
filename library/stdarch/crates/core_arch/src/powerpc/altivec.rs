//! PowerPC AltiVec intrinsics.
//!
//! AltiVec is a brandname trademarked by Freescale (previously Motorola) for
//! the standard `Category:Vector` part of the Power ISA v.2.03 specification.
//! This Category is also known as VMX (used by IBM), and "Velocity Engine" (a
//! brand name previously used by Apple).
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use crate::{
    core_arch::{simd::*, simd_llvm::*},
    mem,
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    /// PowerPC-specific 128-bit wide vector of sixteen packed `i8`
    pub struct vector_signed_char(i8, i8, i8, i8, i8, i8, i8, i8,
                                  i8, i8, i8, i8, i8, i8, i8, i8);
    /// PowerPC-specific 128-bit wide vector of sixteen packed `u8`
    pub struct vector_unsigned_char(u8, u8, u8, u8, u8, u8, u8, u8,
                                    u8, u8, u8, u8, u8, u8, u8, u8);

    /// PowerPC-specific 128-bit wide vector mask of sixteen packed elements
    pub struct vector_bool_char(i8, i8, i8, i8, i8, i8, i8, i8,
                                i8, i8, i8, i8, i8, i8, i8, i8);
    /// PowerPC-specific 128-bit wide vector of eight packed `i16`
    pub struct vector_signed_short(i16, i16, i16, i16, i16, i16, i16, i16);
    /// PowerPC-specific 128-bit wide vector of eight packed `u16`
    pub struct vector_unsigned_short(u16, u16, u16, u16, u16, u16, u16, u16);
    /// PowerPC-specific 128-bit wide vector mask of eight packed elements
    pub struct vector_bool_short(i16, i16, i16, i16, i16, i16, i16, i16);
    // pub struct vector_pixel(???);
    /// PowerPC-specific 128-bit wide vector of four packed `i32`
    pub struct vector_signed_int(i32, i32, i32, i32);
    /// PowerPC-specific 128-bit wide vector of four packed `u32`
    pub struct vector_unsigned_int(u32, u32, u32, u32);
    /// PowerPC-specific 128-bit wide vector mask of four packed elements
    pub struct vector_bool_int(i32, i32, i32, i32);
    /// PowerPC-specific 128-bit wide vector of four packed `f32`
    pub struct vector_float(f32, f32, f32, f32);
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.ppc.altivec.lvx"]
    fn lvx(p: *const i8) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.lvebx"]
    fn lvebx(p: *const i8) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.lvehx"]
    fn lvehx(p: *const i8) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.lvewx"]
    fn lvewx(p: *const i8) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.lvxl"]
    fn lvxl(p: *const i8) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vperm"]
    fn vperm(
        a: vector_signed_int,
        b: vector_signed_int,
        c: vector_unsigned_char,
    ) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmhaddshs"]
    fn vmhaddshs(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_short,
    ) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vmhraddshs"]
    fn vmhraddshs(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_short,
    ) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vmsumuhs"]
    fn vmsumuhs(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vmsumshs"]
    fn vmsumshs(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_int,
    ) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmsumubm"]
    fn vmsumubm(
        a: vector_unsigned_char,
        b: vector_unsigned_char,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vmsummbm"]
    fn vmsummbm(
        a: vector_signed_char,
        b: vector_unsigned_char,
        c: vector_signed_int,
    ) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmsumuhm"]
    fn vmsumuhm(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vmsumshm"]
    fn vmsumshm(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_int,
    ) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmaddfp"]
    fn vmaddfp(a: vector_float, b: vector_float, c: vector_float) -> vector_float;
    #[link_name = "llvm.ppc.altivec.vnmsubfp"]
    fn vnmsubfp(a: vector_float, b: vector_float, c: vector_float) -> vector_float;
    #[link_name = "llvm.ppc.altivec.vsum2sws"]
    fn vsum2sws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vsum4ubs"]
    fn vsum4ubs(a: vector_unsigned_char, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vsum4sbs"]
    fn vsum4sbs(a: vector_signed_char, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vsum4shs"]
    fn vsum4shs(a: vector_signed_short, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmuleub"]
    fn vmuleub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vmulesb"]
    fn vmulesb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vmuleuh"]
    fn vmuleuh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vmulesh"]
    fn vmulesh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vmuloub"]
    fn vmuloub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vmulosb"]
    fn vmulosb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vmulouh"]
    fn vmulouh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int;
    #[link_name = "llvm.ppc.altivec.vmulosh"]
    fn vmulosh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vmaxsb"]
    fn vmaxsb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vmaxsh"]
    fn vmaxsh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vmaxsw"]
    fn vmaxsw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vmaxub"]
    fn vmaxub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vmaxuh"]
    fn vmaxuh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vmaxuw"]
    fn vmaxuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vminsb"]
    fn vminsb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vminsh"]
    fn vminsh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vminsw"]
    fn vminsw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vminub"]
    fn vminub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vminuh"]
    fn vminuh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vminuw"]
    fn vminuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vsubsbs"]
    fn vsubsbs(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vsubshs"]
    fn vsubshs(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vsubsws"]
    fn vsubsws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vsububs"]
    fn vsububs(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vsubuhs"]
    fn vsubuhs(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vsubuws"]
    fn vsubuws(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vaddcuw"]
    fn vaddcuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vaddsbs"]
    fn vaddsbs(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vaddshs"]
    fn vaddshs(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vaddsws"]
    fn vaddsws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vaddubs"]
    fn vaddubs(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vadduhs"]
    fn vadduhs(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vadduws"]
    fn vadduws(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vavgsb"]
    fn vavgsb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vavgsh"]
    fn vavgsh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vavgsw"]
    fn vavgsw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vavgub"]
    fn vavgub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vavguh"]
    fn vavguh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vavguw"]
    fn vavguw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.ceil.v4f32"]
    fn vceil(a: vector_float) -> vector_float;

    #[link_name = "llvm.ppc.altivec.vcmpbfp"]
    fn vcmpbfp(a: vector_float, b: vector_float) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.vcmpequb"]
    fn vcmpequb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_bool_char;
    #[link_name = "llvm.ppc.altivec.vcmpequh"]
    fn vcmpequh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_bool_short;
    #[link_name = "llvm.ppc.altivec.vcmpequw"]
    fn vcmpequw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_bool_int;

    #[link_name = "llvm.ppc.altivec.vcmpgefp"]
    fn vcmpgefp(a: vector_float, b: vector_float) -> vector_bool_int;

    #[link_name = "llvm.ppc.altivec.vcmpgtub"]
    fn vcmpgtub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_bool_char;
    #[link_name = "llvm.ppc.altivec.vcmpgtuh"]
    fn vcmpgtuh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_bool_short;
    #[link_name = "llvm.ppc.altivec.vcmpgtuw"]
    fn vcmpgtuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_bool_int;

    #[link_name = "llvm.ppc.altivec.vcmpgtsb"]
    fn vcmpgtsb(a: vector_signed_char, b: vector_signed_char) -> vector_bool_char;
    #[link_name = "llvm.ppc.altivec.vcmpgtsh"]
    fn vcmpgtsh(a: vector_signed_short, b: vector_signed_short) -> vector_bool_short;
    #[link_name = "llvm.ppc.altivec.vcmpgtsw"]
    fn vcmpgtsw(a: vector_signed_int, b: vector_signed_int) -> vector_bool_int;

    #[link_name = "llvm.ppc.altivec.vexptefp"]
    fn vexptefp(a: vector_float) -> vector_float;

    #[link_name = "llvm.floor.v4f32"]
    fn vfloor(a: vector_float) -> vector_float;

    #[link_name = "llvm.ppc.altivec.vcmpequb.p"]
    fn vcmpequb_p(cr: i32, a: vector_unsigned_char, b: vector_unsigned_char) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpequh.p"]
    fn vcmpequh_p(cr: i32, a: vector_unsigned_short, b: vector_unsigned_short) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpequw.p"]
    fn vcmpequw_p(cr: i32, a: vector_unsigned_int, b: vector_unsigned_int) -> i32;

    #[link_name = "llvm.ppc.altivec.vcmpeqfp.p"]
    fn vcmpeqfp_p(cr: i32, a: vector_float, b: vector_float) -> i32;

    #[link_name = "llvm.ppc.altivec.vcmpgtub.p"]
    fn vcmpgtub_p(cr: i32, a: vector_unsigned_char, b: vector_unsigned_char) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtuh.p"]
    fn vcmpgtuh_p(cr: i32, a: vector_unsigned_short, b: vector_unsigned_short) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtuw.p"]
    fn vcmpgtuw_p(cr: i32, a: vector_unsigned_int, b: vector_unsigned_int) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtsb.p"]
    fn vcmpgtsb_p(cr: i32, a: vector_signed_char, b: vector_signed_char) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtsh.p"]
    fn vcmpgtsh_p(cr: i32, a: vector_signed_short, b: vector_signed_short) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtsw.p"]
    fn vcmpgtsw_p(cr: i32, a: vector_signed_int, b: vector_signed_int) -> i32;

    #[link_name = "llvm.ppc.altivec.vcmpgefp.p"]
    fn vcmpgefp_p(cr: i32, a: vector_float, b: vector_float) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpgtfp.p"]
    fn vcmpgtfp_p(cr: i32, a: vector_float, b: vector_float) -> i32;
    #[link_name = "llvm.ppc.altivec.vcmpbfp.p"]
    fn vcmpbfp_p(cr: i32, a: vector_float, b: vector_float) -> i32;

    #[link_name = "llvm.ppc.altivec.vcfsx"]
    fn vcfsx(a: vector_signed_int, b: i32) -> vector_float;
    #[link_name = "llvm.ppc.altivec.vcfux"]
    fn vcfux(a: vector_unsigned_int, b: i32) -> vector_float;

    #[link_name = "llvm.ppc.altivec.vctsxs"]
    fn vctsxs(a: vector_float, b: i32) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vctuxs"]
    fn vctuxs(a: vector_float, b: i32) -> vector_unsigned_int;

    #[link_name = "llvm.ppc.altivec.vpkshss"]
    fn vpkshss(a: vector_signed_short, b: vector_signed_short) -> vector_signed_char;
    #[link_name = "llvm.ppc.altivec.vpkshus"]
    fn vpkshus(a: vector_signed_short, b: vector_signed_short) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vpkuhus"]
    fn vpkuhus(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_char;
    #[link_name = "llvm.ppc.altivec.vpkswss"]
    fn vpkswss(a: vector_signed_int, b: vector_signed_int) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vpkswus"]
    fn vpkswus(a: vector_signed_int, b: vector_signed_int) -> vector_unsigned_short;
    #[link_name = "llvm.ppc.altivec.vpkuwus"]
    fn vpkuwus(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_short;

    #[link_name = "llvm.ppc.altivec.vupkhsb"]
    fn vupkhsb(a: vector_signed_char) -> vector_signed_short;
    #[link_name = "llvm.ppc.altivec.vupklsb"]
    fn vupklsb(a: vector_signed_char) -> vector_signed_short;

    #[link_name = "llvm.ppc.altivec.vupkhsh"]
    fn vupkhsh(a: vector_signed_short) -> vector_signed_int;
    #[link_name = "llvm.ppc.altivec.vupklsh"]
    fn vupklsh(a: vector_signed_short) -> vector_signed_int;

    #[link_name = "llvm.ppc.altivec.mfvscr"]
    fn mfvscr() -> vector_unsigned_short;

    #[link_name = "llvm.ppc.altivec.vlogefp"]
    fn vlogefp(a: vector_float) -> vector_float;
}

macro_rules! s_t_l {
    (i32x4) => {
        vector_signed_int
    };
    (i16x8) => {
        vector_signed_short
    };
    (i8x16) => {
        vector_signed_char
    };

    (u32x4) => {
        vector_unsigned_int
    };
    (u16x8) => {
        vector_unsigned_short
    };
    (u8x16) => {
        vector_unsigned_char
    };

    (f32x4) => {
        vector_float
    };
}

macro_rules! t_t_l {
    (i32) => {
        vector_signed_int
    };
    (i16) => {
        vector_signed_short
    };
    (i8) => {
        vector_signed_char
    };

    (u32) => {
        vector_unsigned_int
    };
    (u16) => {
        vector_unsigned_short
    };
    (u8) => {
        vector_unsigned_char
    };

    (f32) => {
        vector_float
    };
}

macro_rules! impl_from {
    ($s: ident) => {
        impl From<$s> for s_t_l!($s) {
            fn from (v: $s) -> Self {
                unsafe {
                    transmute(v)
                }
            }
        }
    };
    ($($s: ident),*) => {
        $(
            impl_from! { $s }
        )*
    };
}

impl_from! { i8x16, u8x16,  i16x8, u16x8, i32x4, u32x4, f32x4 }

macro_rules! impl_neg {
    ($s: ident : $zero: expr) => {
        impl crate::ops::Neg for s_t_l!($s) {
            type Output = s_t_l!($s);
            fn neg(self) -> Self::Output {
                let zero = $s::splat($zero);
                unsafe { transmute(simd_sub(zero, transmute(self))) }
            }
        }
    };
}

impl_neg! { i8x16 : 0 }
impl_neg! { i16x8 : 0 }
impl_neg! { i32x4 : 0 }
impl_neg! { f32x4 : 0f32 }

#[macro_use]
mod sealed {
    use super::*;

    macro_rules! impl_vec_cmp {
        ([$Trait:ident $m:ident] ($b:ident, $h:ident, $w:ident)) => {
            impl_vec_cmp! { [$Trait $m] ($b, $b, $h, $h, $w, $w) }
        };
        ([$Trait:ident $m:ident] ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
            impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, vector_unsigned_char) -> vector_bool_char }
            impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, vector_signed_char) -> vector_bool_char }
            impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, vector_unsigned_short) -> vector_bool_short }
            impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, vector_signed_short) -> vector_bool_short }
            impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, vector_unsigned_int) -> vector_bool_int }
            impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, vector_signed_int) -> vector_bool_int }
        }
    }

    macro_rules! impl_vec_any_all {
        ([$Trait:ident $m:ident] ($b:ident, $h:ident, $w:ident)) => {
            impl_vec_any_all! { [$Trait $m] ($b, $b, $h, $h, $w, $w) }
        };
        ([$Trait:ident $m:ident] ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
            impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, vector_unsigned_char) -> bool }
            impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, vector_signed_char) -> bool }
            impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, vector_unsigned_short) -> bool }
            impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, vector_signed_short) -> bool }
            impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, vector_unsigned_int) -> bool }
            impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, vector_signed_int) -> bool }
        }
    }

    pub trait VectorLd {
        type Result;
        unsafe fn vec_ld(self, off: isize) -> Self::Result;
        unsafe fn vec_ldl(self, off: isize) -> Self::Result;
    }

    macro_rules! impl_vec_ld {
        ($fun:ident $fun_lru:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(test, assert_instr(lvx))]
            pub unsafe fn $fun(off: isize, p: *const $ty) -> t_t_l!($ty) {
                let addr = (p as *const i8).offset(off);
                transmute(lvx(addr))
            }

            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(test, assert_instr(lvxl))]
            pub unsafe fn $fun_lru(off: i32, p: *const $ty) -> t_t_l!($ty) {
                let addr = (p as *const i8).offset(off as isize);
                transmute(lvxl(addr))
            }

            impl VectorLd for *const $ty {
                type Result = t_t_l!($ty);
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_ld(self, off: isize) -> Self::Result {
                    $fun(off, self)
                }
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_ldl(self, off: isize) -> Self::Result {
                    $fun(off, self)
                }
            }
        };
    }

    impl_vec_ld! { vec_ld_u8 vec_ldl_u8 u8 }
    impl_vec_ld! { vec_ld_i8 vec_ldl_i8 i8 }

    impl_vec_ld! { vec_ld_u16 vec_ldl_u16 u16 }
    impl_vec_ld! { vec_ld_i16 vec_ldl_i16 i16 }

    impl_vec_ld! { vec_ld_u32 vec_ldl_u32 u32 }
    impl_vec_ld! { vec_ld_i32 vec_ldl_i32 i32 }

    impl_vec_ld! { vec_ld_f32 vec_ldl_f32 f32 }

    pub trait VectorLde {
        type Result;
        unsafe fn vec_lde(self, a: isize) -> Self::Result;
    }

    macro_rules! impl_vec_lde {
        ($fun:ident $instr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $fun(a: isize, b: *const $ty) -> t_t_l!($ty) {
                let addr = (b as *const i8).offset(a);
                transmute($instr(addr))
            }

            impl VectorLde for *const $ty {
                type Result = t_t_l!($ty);
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_lde(self, a: isize) -> Self::Result {
                    $fun(a, self)
                }
            }
        };
    }

    impl_vec_lde! { vec_lde_u8 lvebx u8 }
    impl_vec_lde! { vec_lde_i8 lvebx i8 }

    impl_vec_lde! { vec_lde_u16 lvehx u16 }
    impl_vec_lde! { vec_lde_i16 lvehx i16 }

    impl_vec_lde! { vec_lde_u32 lvewx u32 }
    impl_vec_lde! { vec_lde_i32 lvewx i32 }

    impl_vec_lde! { vec_lde_f32 lvewx f32 }

    pub trait VectorXl {
        type Result;
        unsafe fn vec_xl(self, a: isize) -> Self::Result;
    }

    macro_rules! impl_vec_xl {
        ($fun:ident $notpwr9:ident / $pwr9:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(
                all(test, not(target_feature = "power9-altivec")),
                assert_instr($notpwr9)
            )]
            #[cfg_attr(all(test, target_feature = "power9-altivec"), assert_instr($pwr9))]
            pub unsafe fn $fun(a: isize, b: *const $ty) -> t_t_l!($ty) {
                let addr = (b as *const u8).offset(a);

                // Workaround ptr::copy_nonoverlapping not being inlined
                extern "rust-intrinsic" {
                    #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
                    #[rustc_nounwind]
                    pub fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
                }

                let mut r = mem::MaybeUninit::uninit();

                copy_nonoverlapping(
                    addr,
                    r.as_mut_ptr() as *mut u8,
                    mem::size_of::<t_t_l!($ty)>(),
                );

                r.assume_init()
            }

            impl VectorXl for *const $ty {
                type Result = t_t_l!($ty);
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_xl(self, a: isize) -> Self::Result {
                    $fun(a, self)
                }
            }
        };
    }

    impl_vec_xl! { vec_xl_i8 lxvd2x / lxv i8 }
    impl_vec_xl! { vec_xl_u8 lxvd2x / lxv u8 }
    impl_vec_xl! { vec_xl_i16 lxvd2x / lxv i16 }
    impl_vec_xl! { vec_xl_u16 lxvd2x / lxv u16 }
    impl_vec_xl! { vec_xl_i32 lxvd2x / lxv i32 }
    impl_vec_xl! { vec_xl_u32 lxvd2x / lxv u32 }
    impl_vec_xl! { vec_xl_f32 lxvd2x / lxv f32 }

    test_impl! { vec_floor(a: vector_float) -> vector_float [ vfloor, vrfim / xvrspim ] }

    test_impl! { vec_vexptefp(a: vector_float) -> vector_float [ vexptefp, vexptefp ] }

    test_impl! { vec_vcmpgtub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_bool_char [ vcmpgtub, vcmpgtub ] }
    test_impl! { vec_vcmpgtuh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_bool_short [ vcmpgtuh, vcmpgtuh ] }
    test_impl! { vec_vcmpgtuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_bool_int [ vcmpgtuw, vcmpgtuw ] }

    test_impl! { vec_vcmpgtsb(a: vector_signed_char, b: vector_signed_char) -> vector_bool_char [ vcmpgtsb, vcmpgtsb ] }
    test_impl! { vec_vcmpgtsh(a: vector_signed_short, b: vector_signed_short) -> vector_bool_short [ vcmpgtsh, vcmpgtsh ] }
    test_impl! { vec_vcmpgtsw(a: vector_signed_int, b: vector_signed_int) -> vector_bool_int [ vcmpgtsw, vcmpgtsw ] }

    pub trait VectorCmpGt<Other> {
        type Result;
        unsafe fn vec_cmpgt(self, b: Other) -> Self::Result;
    }

    impl_vec_cmp! { [VectorCmpGt vec_cmpgt] ( vec_vcmpgtub, vec_vcmpgtsb, vec_vcmpgtuh, vec_vcmpgtsh, vec_vcmpgtuw, vec_vcmpgtsw ) }

    test_impl! { vec_vcmpgefp(a: vector_float, b: vector_float) -> vector_bool_int [ vcmpgefp, vcmpgefp ] }

    test_impl! { vec_vcmpequb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_bool_char [ vcmpequb, vcmpequb ] }
    test_impl! { vec_vcmpequh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_bool_short [ vcmpequh, vcmpequh ] }
    test_impl! { vec_vcmpequw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_bool_int [ vcmpequw, vcmpequw ] }

    pub trait VectorCmpEq<Other> {
        type Result;
        unsafe fn vec_cmpeq(self, b: Other) -> Self::Result;
    }

    impl_vec_cmp! { [VectorCmpEq vec_cmpeq] (vec_vcmpequb, vec_vcmpequh, vec_vcmpequw) }

    test_impl! { vec_vcmpbfp(a: vector_float, b: vector_float) -> vector_signed_int [vcmpbfp, vcmpbfp] }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequb.))]
    unsafe fn vcmpequb_all(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpequb_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequb.))]
    unsafe fn vcmpequb_any(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpequb_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequh.))]
    unsafe fn vcmpequh_all(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpequh_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequh.))]
    unsafe fn vcmpequh_any(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpequh_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequw.))]
    unsafe fn vcmpequw_all(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpequw_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequw.))]
    unsafe fn vcmpequw_any(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpequw_p(1, a, b) != 0
    }

    pub trait VectorAllEq<Other> {
        type Result;
        unsafe fn vec_all_eq(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAllEq vec_all_eq] (vcmpequb_all, vcmpequh_all, vcmpequw_all) }

    // TODO: vsx encoding
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpeqfp.))]
    unsafe fn vcmpeqfp_all(a: vector_float, b: vector_float) -> bool {
        vcmpeqfp_p(2, a, b) != 0
    }

    impl VectorAllEq<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_all_eq(self, b: vector_float) -> Self::Result {
            vcmpeqfp_all(self, b)
        }
    }

    pub trait VectorAnyEq<Other> {
        type Result;
        unsafe fn vec_any_eq(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAnyEq vec_any_eq] (vcmpequb_any, vcmpequh_any, vcmpequw_any) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpeqfp.))]
    unsafe fn vcmpeqfp_any(a: vector_float, b: vector_float) -> bool {
        vcmpeqfp_p(1, a, b) != 0
    }

    impl VectorAnyEq<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_any_eq(self, b: vector_float) -> Self::Result {
            vcmpeqfp_any(self, b)
        }
    }

    // All/Any GreaterEqual

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsb.))]
    unsafe fn vcmpgesb_all(a: vector_signed_char, b: vector_signed_char) -> bool {
        vcmpgtsb_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsb.))]
    unsafe fn vcmpgesb_any(a: vector_signed_char, b: vector_signed_char) -> bool {
        vcmpgtsb_p(3, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsh.))]
    unsafe fn vcmpgesh_all(a: vector_signed_short, b: vector_signed_short) -> bool {
        vcmpgtsh_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsh.))]
    unsafe fn vcmpgesh_any(a: vector_signed_short, b: vector_signed_short) -> bool {
        vcmpgtsh_p(3, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsw.))]
    unsafe fn vcmpgesw_all(a: vector_signed_int, b: vector_signed_int) -> bool {
        vcmpgtsw_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsw.))]
    unsafe fn vcmpgesw_any(a: vector_signed_int, b: vector_signed_int) -> bool {
        vcmpgtsw_p(3, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtub.))]
    unsafe fn vcmpgeub_all(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpgtub_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtub.))]
    unsafe fn vcmpgeub_any(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpgtub_p(3, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuh.))]
    unsafe fn vcmpgeuh_all(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpgtuh_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuh.))]
    unsafe fn vcmpgeuh_any(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpgtuh_p(3, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuw.))]
    unsafe fn vcmpgeuw_all(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpgtuw_p(0, b, a) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuw.))]
    unsafe fn vcmpgeuw_any(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpgtuw_p(3, b, a) != 0
    }

    pub trait VectorAllGe<Other> {
        type Result;
        unsafe fn vec_all_ge(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAllGe vec_all_ge] (
        vcmpgeub_all, vcmpgesb_all,
        vcmpgeuh_all, vcmpgesh_all,
        vcmpgeuw_all, vcmpgesw_all
    ) }

    // TODO: vsx encoding
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgefp.))]
    unsafe fn vcmpgefp_all(a: vector_float, b: vector_float) -> bool {
        vcmpgefp_p(2, a, b) != 0
    }

    impl VectorAllGe<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_all_ge(self, b: vector_float) -> Self::Result {
            vcmpgefp_all(self, b)
        }
    }

    pub trait VectorAnyGe<Other> {
        type Result;
        unsafe fn vec_any_ge(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAnyGe vec_any_ge] (
        vcmpgeub_any, vcmpgesb_any,
        vcmpgeuh_any, vcmpgesh_any,
        vcmpgeuw_any, vcmpgesw_any
    ) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgefp.))]
    unsafe fn vcmpgefp_any(a: vector_float, b: vector_float) -> bool {
        vcmpgefp_p(1, a, b) != 0
    }

    impl VectorAnyGe<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_any_ge(self, b: vector_float) -> Self::Result {
            vcmpgefp_any(self, b)
        }
    }

    // All/Any Greater Than

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsb.))]
    unsafe fn vcmpgtsb_all(a: vector_signed_char, b: vector_signed_char) -> bool {
        vcmpgtsb_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsb.))]
    unsafe fn vcmpgtsb_any(a: vector_signed_char, b: vector_signed_char) -> bool {
        vcmpgtsb_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsh.))]
    unsafe fn vcmpgtsh_all(a: vector_signed_short, b: vector_signed_short) -> bool {
        vcmpgtsh_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsh.))]
    unsafe fn vcmpgtsh_any(a: vector_signed_short, b: vector_signed_short) -> bool {
        vcmpgtsh_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsw.))]
    unsafe fn vcmpgtsw_all(a: vector_signed_int, b: vector_signed_int) -> bool {
        vcmpgtsw_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtsw.))]
    unsafe fn vcmpgtsw_any(a: vector_signed_int, b: vector_signed_int) -> bool {
        vcmpgtsw_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtub.))]
    unsafe fn vcmpgtub_all(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpgtub_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtub.))]
    unsafe fn vcmpgtub_any(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpgtub_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuh.))]
    unsafe fn vcmpgtuh_all(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpgtuh_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuh.))]
    unsafe fn vcmpgtuh_any(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpgtuh_p(1, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuw.))]
    unsafe fn vcmpgtuw_all(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpgtuw_p(2, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtuw.))]
    unsafe fn vcmpgtuw_any(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpgtuw_p(1, a, b) != 0
    }

    pub trait VectorAllGt<Other> {
        type Result;
        unsafe fn vec_all_gt(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAllGt vec_all_gt] (
        vcmpgtub_all, vcmpgtsb_all,
        vcmpgtuh_all, vcmpgtsh_all,
        vcmpgtuw_all, vcmpgtsw_all
    ) }

    // TODO: vsx encoding
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtfp.))]
    unsafe fn vcmpgtfp_all(a: vector_float, b: vector_float) -> bool {
        vcmpgtfp_p(2, a, b) != 0
    }

    impl VectorAllGt<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_all_gt(self, b: vector_float) -> Self::Result {
            vcmpgtfp_all(self, b)
        }
    }

    pub trait VectorAnyGt<Other> {
        type Result;
        unsafe fn vec_any_gt(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAnyGt vec_any_gt] (
        vcmpgtub_any, vcmpgtsb_any,
        vcmpgtuh_any, vcmpgtsh_any,
        vcmpgtuw_any, vcmpgtsw_any
    ) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpgtfp.))]
    unsafe fn vcmpgtfp_any(a: vector_float, b: vector_float) -> bool {
        vcmpgtfp_p(1, a, b) != 0
    }

    impl VectorAnyGt<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_any_gt(self, b: vector_float) -> Self::Result {
            vcmpgtfp_any(self, b)
        }
    }

    // All/Any Elements Not Equal

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequb.))]
    unsafe fn vcmpneub_all(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpequb_p(0, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequb.))]
    unsafe fn vcmpneub_any(a: vector_unsigned_char, b: vector_unsigned_char) -> bool {
        vcmpequb_p(3, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequh.))]
    unsafe fn vcmpneuh_all(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpequh_p(0, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequh.))]
    unsafe fn vcmpneuh_any(a: vector_unsigned_short, b: vector_unsigned_short) -> bool {
        vcmpequh_p(3, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequw.))]
    unsafe fn vcmpneuw_all(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpequw_p(0, a, b) != 0
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpequw.))]
    unsafe fn vcmpneuw_any(a: vector_unsigned_int, b: vector_unsigned_int) -> bool {
        vcmpequw_p(3, a, b) != 0
    }

    pub trait VectorAllNe<Other> {
        type Result;
        unsafe fn vec_all_ne(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAllNe vec_all_ne] (vcmpneub_all, vcmpneuh_all, vcmpneuw_all) }

    // TODO: vsx encoding
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpeqfp.))]
    unsafe fn vcmpnefp_all(a: vector_float, b: vector_float) -> bool {
        vcmpeqfp_p(0, a, b) != 0
    }

    impl VectorAllNe<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_all_ne(self, b: vector_float) -> Self::Result {
            vcmpnefp_all(self, b)
        }
    }

    pub trait VectorAnyNe<Other> {
        type Result;
        unsafe fn vec_any_ne(self, b: Other) -> Self::Result;
    }

    impl_vec_any_all! { [VectorAnyNe vec_any_ne] (vcmpneub_any, vcmpneuh_any, vcmpneuw_any) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcmpeqfp.))]
    unsafe fn vcmpnefp_any(a: vector_float, b: vector_float) -> bool {
        vcmpeqfp_p(3, a, b) != 0
    }

    impl VectorAnyNe<vector_float> for vector_float {
        type Result = bool;
        #[inline]
        unsafe fn vec_any_ne(self, b: vector_float) -> Self::Result {
            vcmpnefp_any(self, b)
        }
    }

    test_impl! { vec_vceil(a: vector_float) -> vector_float [vceil, vrfip / xvrspip ] }

    test_impl! { vec_vavgsb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ vavgsb, vavgsb ] }
    test_impl! { vec_vavgsh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [ vavgsh, vavgsh ] }
    test_impl! { vec_vavgsw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [ vavgsw, vavgsw ] }
    test_impl! { vec_vavgub(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [ vavgub, vavgub ] }
    test_impl! { vec_vavguh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [ vavguh, vavguh ] }
    test_impl! { vec_vavguw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [ vavguw, vavguw ] }

    pub trait VectorAvg<Other> {
        type Result;
        unsafe fn vec_avg(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAvg vec_avg] 2 (vec_vavgub, vec_vavgsb, vec_vavguh, vec_vavgsh, vec_vavguw, vec_vavgsw) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, not(target_feature = "vsx")), assert_instr(vandc))]
    #[cfg_attr(all(test, target_feature = "vsx"), assert_instr(xxlandc))]
    unsafe fn andc(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a = transmute(a);
        let b = transmute(b);
        transmute(simd_and(simd_xor(u8x16::splat(0xff), b), a))
    }

    pub trait VectorAndc<Other> {
        type Result;
        unsafe fn vec_andc(self, b: Other) -> Self::Result;
    }

    macro_rules! impl_vec_andc {
        (($a:ty, $b:ty) -> $r:ty) => {
            impl VectorAndc<$b> for $a {
                type Result = $r;
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_andc(self, b: $b) -> Self::Result {
                    transmute(andc(transmute(self), transmute(b)))
                }
            }
        };
        (($a:ty, ~$b:ty) -> $r:ty) => {
            impl_vec_andc! { ($a, $a) -> $r }
            impl_vec_andc! { ($a, $b) -> $r }
            impl_vec_andc! { ($b, $a) -> $r }
        };
    }

    impl_vec_andc! { (vector_unsigned_char, ~vector_bool_char) -> vector_unsigned_char }
    impl_vec_andc! { (vector_signed_char, ~vector_bool_char) -> vector_signed_char }
    impl_vec_andc! { (vector_unsigned_short, ~vector_bool_short) -> vector_unsigned_short }
    impl_vec_andc! { (vector_signed_short, ~vector_bool_short) -> vector_signed_short }
    impl_vec_andc! { (vector_unsigned_int, ~vector_bool_int) -> vector_unsigned_int }
    impl_vec_andc! { (vector_signed_int, ~vector_bool_int) -> vector_signed_int }

    test_impl! { vec_vand(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ simd_and, vand / xxland ] }

    pub trait VectorAnd<Other> {
        type Result;
        unsafe fn vec_and(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAnd vec_and] ~(simd_and) }

    test_impl! { vec_vaddsbs(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ vaddsbs, vaddsbs ] }
    test_impl! { vec_vaddshs(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [ vaddshs, vaddshs ] }
    test_impl! { vec_vaddsws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [ vaddsws, vaddsws ] }
    test_impl! { vec_vaddubs(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [ vaddubs, vaddubs ] }
    test_impl! { vec_vadduhs(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [ vadduhs, vadduhs ] }
    test_impl! { vec_vadduws(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [ vadduws, vadduws ] }

    pub trait VectorAdds<Other> {
        type Result;
        unsafe fn vec_adds(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAdds vec_adds] ~(vaddubs, vaddsbs, vadduhs, vaddshs, vadduws, vaddsws) }

    test_impl! { vec_vaddcuw(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vaddcuw, vaddcuw] }

    test_impl! { vec_vsubsbs(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ vsubsbs, vsubsbs ] }
    test_impl! { vec_vsubshs(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [ vsubshs, vsubshs ] }
    test_impl! { vec_vsubsws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [ vsubsws, vsubsws ] }
    test_impl! { vec_vsububs(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [ vsububs, vsububs ] }
    test_impl! { vec_vsubuhs(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [ vsubuhs, vsubuhs ] }
    test_impl! { vec_vsubuws(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [ vsubuws, vsubuws ] }

    pub trait VectorSubs<Other> {
        type Result;
        unsafe fn vec_subs(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorSubs vec_subs] ~(vsububs, vsubsbs, vsubuhs, vsubshs, vsubuws, vsubsws) }

    pub trait VectorAbs {
        unsafe fn vec_abs(self) -> Self;
    }

    macro_rules! impl_abs {
        ($name:ident,  $ty: ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $name(v: s_t_l!($ty)) -> s_t_l!($ty) {
                v.vec_max(-v)
            }

            impl_vec_trait! { [VectorAbs vec_abs] $name (s_t_l!($ty)) }
        };
    }

    impl_abs! { vec_abs_i8, i8x16 }
    impl_abs! { vec_abs_i16, i16x8 }
    impl_abs! { vec_abs_i32, i32x4 }

    #[inline]
    #[target_feature(enable = "altivec")]
    unsafe fn vec_abs_f32(v: vector_float) -> vector_float {
        let v: u32x4 = transmute(v);

        transmute(simd_and(v, u32x4::splat(0x7FFFFFFF)))
    }

    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f32 (vector_float) }

    pub trait VectorAbss {
        unsafe fn vec_abss(self) -> Self;
    }

    macro_rules! impl_abss {
        ($name:ident,  $ty: ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $name(v: s_t_l!($ty)) -> s_t_l!($ty) {
                let zero: s_t_l!($ty) = transmute(0u8.vec_splats());
                v.vec_max(zero.vec_subs(v))
            }

            impl_vec_trait! { [VectorAbss vec_abss] $name (s_t_l!($ty)) }
        };
    }

    impl_abss! { vec_abss_i8, i8x16 }
    impl_abss! { vec_abss_i16, i16x8 }
    impl_abss! { vec_abss_i32, i32x4 }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vspltb, IMM4 = 15))]
    unsafe fn vspltb<const IMM4: u32>(a: vector_signed_char) -> vector_signed_char {
        static_assert_uimm_bits!(IMM4, 4);
        let b = u8x16::splat(IMM4 as u8);
        vec_perm(a, a, transmute(b))
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vsplth, IMM3 = 7))]
    unsafe fn vsplth<const IMM3: u32>(a: vector_signed_short) -> vector_signed_short {
        static_assert_uimm_bits!(IMM3, 3);
        let b0 = IMM3 as u8 * 2;
        let b1 = b0 + 1;
        let b = u8x16::new(
            b0, b1, b0, b1, b0, b1, b0, b1, b0, b1, b0, b1, b0, b1, b0, b1,
        );
        vec_perm(a, a, transmute(b))
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, not(target_feature = "vsx")), assert_instr(vspltw, IMM2 = 3))]
    #[cfg_attr(all(test, target_feature = "vsx"), assert_instr(xxspltw, IMM2 = 3))]
    unsafe fn vspltw<const IMM2: u32>(a: vector_signed_int) -> vector_signed_int {
        static_assert_uimm_bits!(IMM2, 2);
        let b0 = IMM2 as u8 * 4;
        let b1 = b0 + 1;
        let b2 = b0 + 2;
        let b3 = b0 + 3;
        let b = u8x16::new(
            b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3,
        );
        vec_perm(a, a, transmute(b))
    }

    pub trait VectorSplat {
        unsafe fn vec_splat<const IMM: u32>(self) -> Self;
    }

    macro_rules! impl_vec_splat {
        ($ty:ty, $fun:ident) => {
            impl VectorSplat for $ty {
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_splat<const IMM: u32>(self) -> Self {
                    transmute($fun::<IMM>(transmute(self)))
                }
            }
        };
    }

    impl_vec_splat! { vector_signed_char, vspltb }
    impl_vec_splat! { vector_unsigned_char, vspltb }
    impl_vec_splat! { vector_bool_char, vspltb }
    impl_vec_splat! { vector_signed_short, vsplth }
    impl_vec_splat! { vector_unsigned_short, vsplth }
    impl_vec_splat! { vector_bool_short, vsplth }
    impl_vec_splat! { vector_signed_int, vspltw }
    impl_vec_splat! { vector_unsigned_int, vspltw }
    impl_vec_splat! { vector_bool_int, vspltw }

    macro_rules! splat {
        ($name:ident, $v:ident, $r:ident [$instr:ident, $doc:literal]) => {
            #[doc = $doc]
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(test, assert_instr($instr, IMM5 = 1))]
            pub unsafe fn $name<const IMM5: i8>() -> s_t_l!($r) {
                static_assert_simm_bits!(IMM5, 5);
                transmute($r::splat(IMM5 as $v))
            }
        };
    }

    macro_rules! splats {
        ($name:ident, $v:ident, $r:ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $name(v: $v) -> s_t_l!($r) {
                transmute($r::splat(v))
            }
        };
    }

    splats! { splats_u8, u8, u8x16 }
    splats! { splats_u16, u16, u16x8 }
    splats! { splats_u32, u32, u32x4 }
    splats! { splats_i8, i8, i8x16 }
    splats! { splats_i16, i16, i16x8 }
    splats! { splats_i32, i32, i32x4 }
    splats! { splats_f32, f32, f32x4 }

    test_impl! { vec_splats_u8 (v: u8) -> vector_unsigned_char [splats_u8, vspltb] }
    test_impl! { vec_splats_u16 (v: u16) -> vector_unsigned_short [splats_u16, vsplth] }
    test_impl! { vec_splats_u32 (v: u32) -> vector_unsigned_int [splats_u32, vspltw / xxspltw] }
    test_impl! { vec_splats_i8 (v: i8) -> vector_signed_char [splats_i8, vspltb] }
    test_impl! { vec_splats_i16 (v: i16) -> vector_signed_short [splats_i16, vsplth] }
    test_impl! { vec_splats_i32 (v: i32) -> vector_signed_int [splats_i32, vspltw / xxspltw] }
    test_impl! { vec_splats_f32 (v: f32) -> vector_float [splats_f32, vspltw / xxspltw] }

    pub trait VectorSplats {
        type Result;
        unsafe fn vec_splats(self) -> Self::Result;
    }

    macro_rules! impl_vec_splats {
        ($(($fn:ident ($ty:ty) -> $r:ty)),*) => {
            $(
                impl_vec_trait!{ [VectorSplats vec_splats] $fn ($ty) -> $r }
            )*
        }
    }

    impl_vec_splats! {
        (vec_splats_u8 (u8) -> vector_unsigned_char),
        (vec_splats_i8 (i8) -> vector_signed_char),
        (vec_splats_u16 (u16) -> vector_unsigned_short),
        (vec_splats_i16 (i16) -> vector_signed_short),
        (vec_splats_u32 (u32) -> vector_unsigned_int),
        (vec_splats_i32 (i32) -> vector_signed_int),
        (vec_splats_f32 (f32) -> vector_float)
    }

    test_impl! { vec_vsububm (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [simd_sub, vsububm] }
    test_impl! { vec_vsubuhm (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [simd_sub, vsubuhm] }
    test_impl! { vec_vsubuwm (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [simd_sub, vsubuwm] }

    pub trait VectorSub<Other> {
        type Result;
        unsafe fn vec_sub(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorSub vec_sub] ~(simd_sub, simd_sub, simd_sub, simd_sub, simd_sub, simd_sub) }
    impl_vec_trait! { [VectorSub vec_sub] simd_sub(vector_float, vector_float) -> vector_float }

    test_impl! { vec_vminsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vminsb, vminsb] }
    test_impl! { vec_vminsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vminsh, vminsh] }
    test_impl! { vec_vminsw (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vminsw, vminsw] }

    test_impl! { vec_vminub (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vminub, vminub] }
    test_impl! { vec_vminuh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vminuh, vminuh] }
    test_impl! { vec_vminuw (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vminuw, vminuw] }

    pub trait VectorMin<Other> {
        type Result;
        unsafe fn vec_min(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorMin vec_min] ~(vminub, vminsb, vminuh, vminsh, vminuw, vminsw) }

    test_impl! { vec_vmaxsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vmaxsb, vmaxsb] }
    test_impl! { vec_vmaxsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vmaxsh, vmaxsh] }
    test_impl! { vec_vmaxsw (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vmaxsw, vmaxsw] }

    test_impl! { vec_vmaxub (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vmaxub, vmaxub] }
    test_impl! { vec_vmaxuh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vmaxuh, vmaxuh] }
    test_impl! { vec_vmaxuw (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vmaxuw, vmaxuw] }

    pub trait VectorMax<Other> {
        type Result;
        unsafe fn vec_max(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorMax vec_max] ~(vmaxub, vmaxsb, vmaxuh, vmaxsh, vmaxuw, vmaxsw) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmuleub))]
    unsafe fn vec_vmuleub(
        a: vector_unsigned_char,
        b: vector_unsigned_char,
    ) -> vector_unsigned_short {
        vmuleub(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmulesb))]
    unsafe fn vec_vmulesb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_short {
        vmulesb(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmuleuh))]
    unsafe fn vec_vmuleuh(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
    ) -> vector_unsigned_int {
        vmuleuh(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmulesh))]
    unsafe fn vec_vmulesh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_int {
        vmulesh(a, b)
    }

    pub trait VectorMule<Result> {
        unsafe fn vec_mule(self, b: Self) -> Result;
    }

    impl VectorMule<vector_unsigned_short> for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mule(self, b: Self) -> vector_unsigned_short {
            vmuleub(self, b)
        }
    }
    impl VectorMule<vector_signed_short> for vector_signed_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mule(self, b: Self) -> vector_signed_short {
            vmulesb(self, b)
        }
    }
    impl VectorMule<vector_unsigned_int> for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mule(self, b: Self) -> vector_unsigned_int {
            vmuleuh(self, b)
        }
    }
    impl VectorMule<vector_signed_int> for vector_signed_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mule(self, b: Self) -> vector_signed_int {
            vmulesh(self, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmuloub))]
    unsafe fn vec_vmuloub(
        a: vector_unsigned_char,
        b: vector_unsigned_char,
    ) -> vector_unsigned_short {
        vmuloub(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmulosb))]
    unsafe fn vec_vmulosb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_short {
        vmulosb(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmulouh))]
    unsafe fn vec_vmulouh(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
    ) -> vector_unsigned_int {
        vmulouh(a, b)
    }
    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmulosh))]
    unsafe fn vec_vmulosh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_int {
        vmulosh(a, b)
    }

    pub trait VectorMulo<Result> {
        unsafe fn vec_mulo(self, b: Self) -> Result;
    }

    impl VectorMulo<vector_unsigned_short> for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mulo(self, b: Self) -> vector_unsigned_short {
            vmuloub(self, b)
        }
    }
    impl VectorMulo<vector_signed_short> for vector_signed_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mulo(self, b: Self) -> vector_signed_short {
            vmulosb(self, b)
        }
    }
    impl VectorMulo<vector_unsigned_int> for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mulo(self, b: Self) -> vector_unsigned_int {
            vmulouh(self, b)
        }
    }
    impl VectorMulo<vector_signed_int> for vector_signed_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_mulo(self, b: Self) -> vector_signed_int {
            vmulosh(self, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vsum4ubs))]
    unsafe fn vec_vsum4ubs(a: vector_unsigned_char, b: vector_unsigned_int) -> vector_unsigned_int {
        vsum4ubs(a, b)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vsum4sbs))]
    unsafe fn vec_vsum4sbs(a: vector_signed_char, b: vector_signed_int) -> vector_signed_int {
        vsum4sbs(a, b)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vsum4shs))]
    unsafe fn vec_vsum4shs(a: vector_signed_short, b: vector_signed_int) -> vector_signed_int {
        vsum4shs(a, b)
    }

    pub trait VectorSum4s<Other> {
        unsafe fn vec_sum4s(self, b: Other) -> Other;
    }

    impl VectorSum4s<vector_unsigned_int> for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_sum4s(self, b: vector_unsigned_int) -> vector_unsigned_int {
            vsum4ubs(self, b)
        }
    }

    impl VectorSum4s<vector_signed_int> for vector_signed_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_sum4s(self, b: vector_signed_int) -> vector_signed_int {
            vsum4sbs(self, b)
        }
    }

    impl VectorSum4s<vector_signed_int> for vector_signed_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_sum4s(self, b: vector_signed_int) -> vector_signed_int {
            vsum4shs(self, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vsum2sws))]
    unsafe fn vec_vsum2sws(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        vsum2sws(a, b)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vnmsubfp))]
    unsafe fn vec_vnmsubfp(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
        vnmsubfp(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmaddfp))]
    unsafe fn vec_vmaddfp(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
        vmaddfp(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsumubm))]
    unsafe fn vec_vmsumubm(
        a: vector_unsigned_char,
        b: vector_unsigned_char,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int {
        vmsumubm(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsummbm))]
    unsafe fn vec_vmsummbm(
        a: vector_signed_char,
        b: vector_unsigned_char,
        c: vector_signed_int,
    ) -> vector_signed_int {
        vmsummbm(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsumuhm))]
    unsafe fn vec_vmsumuhm(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int {
        vmsumuhm(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsumshm))]
    unsafe fn vec_vmsumshm(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_int,
    ) -> vector_signed_int {
        vmsumshm(a, b, c)
    }

    pub trait VectorMsum<B, Other> {
        unsafe fn vec_msum(self, b: B, c: Other) -> Other;
    }

    impl VectorMsum<vector_unsigned_char, vector_unsigned_int> for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msum(
            self,
            b: vector_unsigned_char,
            c: vector_unsigned_int,
        ) -> vector_unsigned_int {
            vmsumubm(self, b, c)
        }
    }

    impl VectorMsum<vector_unsigned_char, vector_signed_int> for vector_signed_char {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msum(
            self,
            b: vector_unsigned_char,
            c: vector_signed_int,
        ) -> vector_signed_int {
            vmsummbm(self, b, c)
        }
    }

    impl VectorMsum<vector_unsigned_short, vector_unsigned_int> for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msum(
            self,
            b: vector_unsigned_short,
            c: vector_unsigned_int,
        ) -> vector_unsigned_int {
            vmsumuhm(self, b, c)
        }
    }

    impl VectorMsum<vector_signed_short, vector_signed_int> for vector_signed_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msum(
            self,
            b: vector_signed_short,
            c: vector_signed_int,
        ) -> vector_signed_int {
            vmsumshm(self, b, c)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsumuhs))]
    unsafe fn vec_vmsumuhs(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
        c: vector_unsigned_int,
    ) -> vector_unsigned_int {
        vmsumuhs(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmsumshs))]
    unsafe fn vec_vmsumshs(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_int,
    ) -> vector_signed_int {
        vmsumshs(a, b, c)
    }

    pub trait VectorMsums<Other> {
        unsafe fn vec_msums(self, b: Self, c: Other) -> Other;
    }

    impl VectorMsums<vector_unsigned_int> for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msums(self, b: Self, c: vector_unsigned_int) -> vector_unsigned_int {
            vmsumuhs(self, b, c)
        }
    }

    impl VectorMsums<vector_signed_int> for vector_signed_short {
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_msums(self, b: Self, c: vector_signed_int) -> vector_signed_int {
            vmsumshs(self, b, c)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vperm))]
    unsafe fn vec_vperm(
        a: vector_signed_int,
        b: vector_signed_int,
        c: vector_unsigned_char,
    ) -> vector_signed_int {
        vperm(a, b, c)
    }

    pub trait VectorPerm {
        unsafe fn vec_vperm(self, b: Self, c: vector_unsigned_char) -> Self;
    }

    macro_rules! vector_perm {
        {$impl: ident} => {
            impl VectorPerm for $impl {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn vec_vperm(self, b: Self, c: vector_unsigned_char) -> Self {
                    transmute(vec_vperm(transmute(self), transmute(b), c))
                }
            }
        }
    }

    vector_perm! { vector_signed_char }
    vector_perm! { vector_unsigned_char }
    vector_perm! { vector_bool_char }

    vector_perm! { vector_signed_short }
    vector_perm! { vector_unsigned_short }
    vector_perm! { vector_bool_short }

    vector_perm! { vector_signed_int }
    vector_perm! { vector_unsigned_int }
    vector_perm! { vector_bool_int }

    vector_perm! { vector_float }

    pub trait VectorAdd<Other> {
        type Result;
        unsafe fn vec_add(self, other: Other) -> Self::Result;
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_bc_sc(a: vector_bool_char, b: vector_signed_char) -> vector_signed_char {
        simd_add(transmute(a), b)
    }
    impl VectorAdd<vector_signed_char> for vector_bool_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_char) -> Self::Result {
            vec_add_bc_sc(self, other)
        }
    }
    impl VectorAdd<vector_bool_char> for vector_signed_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_char) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_sc_sc(
        a: vector_signed_char,
        b: vector_signed_char,
    ) -> vector_signed_char {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_char> for vector_signed_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_char) -> Self::Result {
            vec_add_sc_sc(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_bc_uc(
        a: vector_bool_char,
        b: vector_unsigned_char,
    ) -> vector_unsigned_char {
        simd_add(transmute(a), b)
    }
    impl VectorAdd<vector_unsigned_char> for vector_bool_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_char) -> Self::Result {
            vec_add_bc_uc(self, other)
        }
    }
    impl VectorAdd<vector_bool_char> for vector_unsigned_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_char) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_uc_uc(
        a: vector_unsigned_char,
        b: vector_unsigned_char,
    ) -> vector_unsigned_char {
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_char> for vector_unsigned_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_char) -> Self::Result {
            vec_add_uc_uc(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_bs_ss(
        a: vector_bool_short,
        b: vector_signed_short,
    ) -> vector_signed_short {
        let a: i16x8 = transmute(a);
        let a: vector_signed_short = simd_cast(a);
        simd_add(a, b)
    }

    impl VectorAdd<vector_signed_short> for vector_bool_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_short) -> Self::Result {
            vec_add_bs_ss(self, other)
        }
    }
    impl VectorAdd<vector_bool_short> for vector_signed_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_short) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_ss_ss(
        a: vector_signed_short,
        b: vector_signed_short,
    ) -> vector_signed_short {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_short> for vector_signed_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_short) -> Self::Result {
            vec_add_ss_ss(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_bs_us(
        a: vector_bool_short,
        b: vector_unsigned_short,
    ) -> vector_unsigned_short {
        let a: i16x8 = transmute(a);
        let a: vector_unsigned_short = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_short> for vector_bool_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_short) -> Self::Result {
            vec_add_bs_us(self, other)
        }
    }
    impl VectorAdd<vector_bool_short> for vector_unsigned_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_short) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_us_us(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
    ) -> vector_unsigned_short {
        simd_add(a, b)
    }

    impl VectorAdd<vector_unsigned_short> for vector_unsigned_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_short) -> Self::Result {
            vec_add_us_us(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_bi_si(a: vector_bool_int, b: vector_signed_int) -> vector_signed_int {
        let a: i32x4 = transmute(a);
        let a: vector_signed_int = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_int> for vector_bool_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_int) -> Self::Result {
            vec_add_bi_si(self, other)
        }
    }
    impl VectorAdd<vector_bool_int> for vector_signed_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_int) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_si_si(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_int> for vector_signed_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_int) -> Self::Result {
            vec_add_si_si(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_bi_ui(a: vector_bool_int, b: vector_unsigned_int) -> vector_unsigned_int {
        let a: i32x4 = transmute(a);
        let a: vector_unsigned_int = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_int> for vector_bool_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_int) -> Self::Result {
            vec_add_bi_ui(self, other)
        }
    }
    impl VectorAdd<vector_bool_int> for vector_unsigned_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_int) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_ui_ui(
        a: vector_unsigned_int,
        b: vector_unsigned_int,
    ) -> vector_unsigned_int {
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_int> for vector_unsigned_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_int) -> Self::Result {
            vec_add_ui_ui(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(xvaddsp))]
    pub unsafe fn vec_add_float_float(a: vector_float, b: vector_float) -> vector_float {
        simd_add(a, b)
    }

    impl VectorAdd<vector_float> for vector_float {
        type Result = vector_float;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_float) -> Self::Result {
            vec_add_float_float(self, other)
        }
    }

    pub trait VectorMladd<Other> {
        type Result;
        unsafe fn vec_mladd(self, b: Other, c: Other) -> Self::Result;
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vmladduhm))]
    unsafe fn mladd(
        a: vector_signed_short,
        b: vector_signed_short,
        c: vector_signed_short,
    ) -> vector_signed_short {
        let a: i16x8 = transmute(a);
        let b: i16x8 = transmute(b);
        let c: i16x8 = transmute(c);
        transmute(simd_add(simd_mul(a, b), c))
    }

    macro_rules! vector_mladd {
        ($a: ident, $bc: ident, $d: ident) => {
            impl VectorMladd<$bc> for $a {
                type Result = $d;
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mladd(self, b: $bc, c: $bc) -> Self::Result {
                    let a = transmute(self);
                    let b = transmute(b);
                    let c = transmute(c);

                    transmute(mladd(a, b, c))
                }
            }
        };
    }

    vector_mladd! { vector_unsigned_short, vector_unsigned_short, vector_unsigned_short }
    vector_mladd! { vector_unsigned_short, vector_signed_short, vector_signed_short }
    vector_mladd! { vector_signed_short, vector_unsigned_short, vector_signed_short }
    vector_mladd! { vector_signed_short, vector_signed_short, vector_signed_short }

    pub trait VectorOr<Other> {
        type Result;
        unsafe fn vec_or(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorOr vec_or] ~(simd_or) }

    pub trait VectorXor<Other> {
        type Result;
        unsafe fn vec_xor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorXor vec_xor] ~(simd_xor) }

    macro_rules! vector_vnor {
        ($fun:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(all(test, not(target_feature = "vsx")), assert_instr(vnor))]
            #[cfg_attr(all(test, target_feature = "vsx"), assert_instr(xxlnor))]
            pub unsafe fn $fun(a: t_t_l!($ty), b: t_t_l!($ty)) -> t_t_l!($ty) {
                let o = vec_splats(!0 as $ty);
                vec_xor(vec_or(a, b), o)
            }
        };
    }

    vector_vnor! { vec_vnorsb i8 }
    vector_vnor! { vec_vnorsh i16 }
    vector_vnor! { vec_vnorsw i32 }
    vector_vnor! { vec_vnorub u8 }
    vector_vnor! { vec_vnoruh u16 }
    vector_vnor! { vec_vnoruw u32 }

    pub trait VectorNor<Other> {
        type Result;
        unsafe fn vec_nor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorNor vec_nor] 2 (vec_vnorub, vec_vnorsb, vec_vnoruh, vec_vnorsh, vec_vnoruw, vec_vnorsw) }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcfsx, IMM5 = 1))]
    unsafe fn vec_ctf_i32<const IMM5: i32>(a: vector_signed_int) -> vector_float {
        static_assert_uimm_bits!(IMM5, 5);
        vcfsx(a, IMM5)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vcfux, IMM5 = 1))]
    unsafe fn vec_ctf_u32<const IMM5: i32>(a: vector_unsigned_int) -> vector_float {
        static_assert_uimm_bits!(IMM5, 5);
        vcfux(a, IMM5)
    }

    pub trait VectorCtf {
        unsafe fn vec_ctf<const IMM5: i32>(self) -> vector_float;
    }

    impl VectorCtf for vector_signed_int {
        unsafe fn vec_ctf<const IMM5: i32>(self) -> vector_float {
            vec_ctf_i32::<IMM5>(self)
        }
    }

    impl VectorCtf for vector_unsigned_int {
        unsafe fn vec_ctf<const IMM5: i32>(self) -> vector_float {
            vec_ctf_u32::<IMM5>(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(vmrghb))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(vmrglb))]
    unsafe fn vec_vmrglb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let mergel_perm = transmute(u8x16::new(
            0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E,
            0x0F, 0x1F,
        ));
        vec_perm(a, b, mergel_perm)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(vmrghh))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(vmrglh))]
    unsafe fn vec_vmrglh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short {
        let mergel_perm = transmute(u8x16::new(
            0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B, 0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F,
            0x1E, 0x1F,
        ));
        vec_perm(a, b, mergel_perm)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", not(target_feature = "vsx")),
        assert_instr(vmrghw)
    )]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "vsx"),
        assert_instr(xxmrghw)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", not(target_feature = "vsx")),
        assert_instr(vmrglw)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "vsx"),
        assert_instr(xxmrglw)
    )]
    unsafe fn vec_vmrglw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let mergel_perm = transmute(u8x16::new(
            0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D,
            0x1E, 0x1F,
        ));
        vec_perm(a, b, mergel_perm)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(vmrglb))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(vmrghb))]
    unsafe fn vec_vmrghb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let mergel_perm = transmute(u8x16::new(
            0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14, 0x05, 0x15, 0x06, 0x16,
            0x07, 0x17,
        ));
        vec_perm(a, b, mergel_perm)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(vmrglh))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(vmrghh))]
    unsafe fn vec_vmrghh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short {
        let mergel_perm = transmute(u8x16::new(
            0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13, 0x04, 0x05, 0x14, 0x15, 0x06, 0x07,
            0x16, 0x17,
        ));
        vec_perm(a, b, mergel_perm)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", not(target_feature = "vsx")),
        assert_instr(vmrglw)
    )]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "vsx"),
        assert_instr(xxmrglw)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", not(target_feature = "vsx")),
        assert_instr(vmrghw)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "vsx"),
        assert_instr(xxmrghw)
    )]
    unsafe fn vec_vmrghw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let mergel_perm = transmute(u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x04, 0x05, 0x06, 0x07, 0x14, 0x15,
            0x16, 0x17,
        ));
        vec_perm(a, b, mergel_perm)
    }

    pub trait VectorMergeh<Other> {
        type Result;
        unsafe fn vec_mergeh(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorMergeh vec_mergeh]+ 2b (vec_vmrghb, vec_vmrghh, vec_vmrghw) }
    impl_vec_trait! { [VectorMergeh vec_mergeh]+ vec_vmrghw (vector_float, vector_float) -> vector_float }

    pub trait VectorMergel<Other> {
        type Result;
        unsafe fn vec_mergel(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorMergel vec_mergel]+ 2b (vec_vmrglb, vec_vmrglh, vec_vmrglw) }
    impl_vec_trait! { [VectorMergel vec_mergel]+ vec_vmrglw (vector_float, vector_float) -> vector_float }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkuhum))]
    unsafe fn vec_vpkuhum(a: vector_signed_short, b: vector_signed_short) -> vector_signed_char {
        let pack_perm = if cfg!(target_endian = "little") {
            transmute(u8x16::new(
                0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A,
                0x1C, 0x1E,
            ))
        } else {
            transmute(u8x16::new(
                0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1B,
                0x1D, 0x1F,
            ))
        };

        transmute(vec_perm(a, b, pack_perm))
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkuwum))]
    unsafe fn vec_vpkuwum(a: vector_signed_int, b: vector_signed_int) -> vector_signed_short {
        let pack_perm = if cfg!(target_endian = "little") {
            transmute(u8x16::new(
                0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D, 0x10, 0x11, 0x14, 0x15, 0x18, 0x19,
                0x1C, 0x1D,
            ))
        } else {
            transmute(u8x16::new(
                0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F, 0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B,
                0x1E, 0x1F,
            ))
        };

        transmute(vec_perm(a, b, pack_perm))
    }

    pub trait VectorPack<Other> {
        type Result;
        unsafe fn vec_pack(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuhum (vector_signed_short, vector_signed_short) -> vector_signed_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuhum (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuhum (vector_bool_short, vector_bool_short) -> vector_bool_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuwum (vector_signed_int, vector_signed_int) -> vector_signed_short }
    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuwum (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPack vec_pack]+ vec_vpkuwum (vector_bool_int, vector_bool_int) -> vector_bool_short }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkshss))]
    unsafe fn vec_vpkshss(a: vector_signed_short, b: vector_signed_short) -> vector_signed_char {
        if cfg!(target_endian = "little") {
            vpkshss(b, a)
        } else {
            vpkshss(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkshus))]
    unsafe fn vec_vpkshus(a: vector_signed_short, b: vector_signed_short) -> vector_unsigned_char {
        if cfg!(target_endian = "little") {
            vpkshus(b, a)
        } else {
            vpkshus(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkuhus))]
    unsafe fn vec_vpkuhus(
        a: vector_unsigned_short,
        b: vector_unsigned_short,
    ) -> vector_unsigned_char {
        if cfg!(target_endian = "little") {
            vpkuhus(b, a)
        } else {
            vpkuhus(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkswss))]
    unsafe fn vec_vpkswss(a: vector_signed_int, b: vector_signed_int) -> vector_signed_short {
        if cfg!(target_endian = "little") {
            vpkswss(b, a)
        } else {
            vpkswss(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkswus))]
    unsafe fn vec_vpkswus(a: vector_signed_int, b: vector_signed_int) -> vector_unsigned_short {
        if cfg!(target_endian = "little") {
            vpkswus(b, a)
        } else {
            vpkswus(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vpkuwus))]
    unsafe fn vec_vpkuwus(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_short {
        if cfg!(target_endian = "little") {
            vpkuwus(b, a)
        } else {
            vpkuwus(a, b)
        }
    }

    pub trait VectorPacks<Other> {
        type Result;
        unsafe fn vec_packs(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorPacks vec_packs] vec_vpkshss (vector_signed_short, vector_signed_short) -> vector_signed_char }
    impl_vec_trait! { [VectorPacks vec_packs] vec_vpkuhus (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacks vec_packs] vec_vpkswss (vector_signed_int, vector_signed_int) -> vector_signed_short }
    impl_vec_trait! { [VectorPacks vec_packs] vec_vpkuwus (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }

    pub trait VectorPacksu<Other> {
        type Result;
        unsafe fn vec_packsu(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorPacksu vec_packsu] vec_vpkshus (vector_signed_short, vector_signed_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacksu vec_packsu] vec_vpkuhus (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacksu vec_packsu] vec_vpkswus (vector_signed_int, vector_signed_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPacksu vec_packsu] vec_vpkuwus (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }

    macro_rules! impl_vec_unpack {
        ($fun:ident ($a:ident) -> $r:ident [$little:ident, $big:ident]) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(all(test, target_endian = "little"), assert_instr($little))]
            #[cfg_attr(all(test, target_endian = "big"), assert_instr($big))]
            unsafe fn $fun(a: $a) -> $r {
                if cfg!(target_endian = "little") {
                    $little(a)
                } else {
                    $big(a)
                }
            }
        };
    }

    impl_vec_unpack! { vec_vupkhsb (vector_signed_char) -> vector_signed_short [vupklsb, vupkhsb] }
    impl_vec_unpack! { vec_vupklsb (vector_signed_char) -> vector_signed_short [vupkhsb, vupklsb] }
    impl_vec_unpack! { vec_vupkhsh (vector_signed_short) -> vector_signed_int [vupklsh, vupkhsh] }
    impl_vec_unpack! { vec_vupklsh (vector_signed_short) -> vector_signed_int [vupkhsh, vupklsh] }

    pub trait VectorUnpackh {
        type Result;
        unsafe fn vec_unpackh(self) -> Self::Result;
    }

    impl_vec_trait! { [VectorUnpackh vec_unpackh] vec_vupkhsb (vector_signed_char) -> vector_signed_short }
    impl_vec_trait! { [VectorUnpackh vec_unpackh]+ vec_vupkhsb (vector_bool_char) -> vector_bool_short }
    impl_vec_trait! { [VectorUnpackh vec_unpackh] vec_vupkhsh (vector_signed_short) -> vector_signed_int }
    impl_vec_trait! { [VectorUnpackh vec_unpackh]+ vec_vupkhsh (vector_bool_short) -> vector_bool_int }

    pub trait VectorUnpackl {
        type Result;
        unsafe fn vec_unpackl(self) -> Self::Result;
    }

    impl_vec_trait! { [VectorUnpackl vec_unpackl] vec_vupklsb (vector_signed_char) -> vector_signed_short }
    impl_vec_trait! { [VectorUnpackl vec_unpackl]+ vec_vupklsb (vector_bool_char) -> vector_bool_short }
    impl_vec_trait! { [VectorUnpackl vec_unpackl] vec_vupklsh (vector_signed_short) -> vector_signed_int }
    impl_vec_trait! { [VectorUnpackl vec_unpackl]+ vec_vupklsh (vector_bool_short) -> vector_bool_int }
}

/// Vector Merge Low
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_mergel<T, U>(a: T, b: U) -> <T as sealed::VectorMergel<U>>::Result
where
    T: sealed::VectorMergel<U>,
{
    a.vec_mergel(b)
}

/// Vector Merge High
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_mergeh<T, U>(a: T, b: U) -> <T as sealed::VectorMergeh<U>>::Result
where
    T: sealed::VectorMergeh<U>,
{
    a.vec_mergeh(b)
}

/// Vector Pack
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_pack<T, U>(a: T, b: U) -> <T as sealed::VectorPack<U>>::Result
where
    T: sealed::VectorPack<U>,
{
    a.vec_pack(b)
}

/// Vector Pack Saturated
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_packs<T, U>(a: T, b: U) -> <T as sealed::VectorPacks<U>>::Result
where
    T: sealed::VectorPacks<U>,
{
    a.vec_packs(b)
}

/// Vector Pack Saturated Unsigned
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_packsu<T, U>(a: T, b: U) -> <T as sealed::VectorPacksu<U>>::Result
where
    T: sealed::VectorPacksu<U>,
{
    a.vec_packsu(b)
}

/// Vector Unpack High
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_unpackh<T>(a: T) -> <T as sealed::VectorUnpackh>::Result
where
    T: sealed::VectorUnpackh,
{
    a.vec_unpackh()
}

/// Vector Unpack Low
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_unpackl<T>(a: T) -> <T as sealed::VectorUnpackl>::Result
where
    T: sealed::VectorUnpackl,
{
    a.vec_unpackl()
}

/// Vector Load Indexed.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_ld<T>(off: isize, p: T) -> <T as sealed::VectorLd>::Result
where
    T: sealed::VectorLd,
{
    p.vec_ld(off)
}

/// Vector Load Indexed Least Recently Used.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_ldl<T>(off: isize, p: T) -> <T as sealed::VectorLd>::Result
where
    T: sealed::VectorLd,
{
    p.vec_ldl(off)
}

/// Vector Load Element Indexed.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_lde<T>(off: isize, p: T) -> <T as sealed::VectorLde>::Result
where
    T: sealed::VectorLde,
{
    p.vec_lde(off)
}

/// VSX Unaligned Load
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_xl<T>(off: isize, p: T) -> <T as sealed::VectorXl>::Result
where
    T: sealed::VectorXl,
{
    p.vec_xl(off)
}

/// Vector Base-2 Logarithm Estimate
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(vlogefp))]
pub unsafe fn vec_loge(a: vector_float) -> vector_float {
    vlogefp(a)
}

/// Vector floor.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_floor(a: vector_float) -> vector_float {
    sealed::vec_floor(a)
}

/// Vector expte.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_expte(a: vector_float) -> vector_float {
    sealed::vec_vexptefp(a)
}

/// Vector cmplt.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmplt<T, U>(a: U, b: T) -> <T as sealed::VectorCmpGt<U>>::Result
where
    T: sealed::VectorCmpGt<U>,
{
    vec_cmpgt(b, a)
}

/// Vector cmple.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmple(a: vector_float, b: vector_float) -> vector_bool_int {
    vec_cmpge(b, a)
}

/// Vector cmpgt.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmpgt<T, U>(a: T, b: U) -> <T as sealed::VectorCmpGt<U>>::Result
where
    T: sealed::VectorCmpGt<U>,
{
    a.vec_cmpgt(b)
}

/// Vector cmpge.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmpge(a: vector_float, b: vector_float) -> vector_bool_int {
    sealed::vec_vcmpgefp(a, b)
}

/// Vector cmpeq.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmpeq<T, U>(a: T, b: U) -> <T as sealed::VectorCmpEq<U>>::Result
where
    T: sealed::VectorCmpEq<U>,
{
    a.vec_cmpeq(b)
}

/// Vector cmpb.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_cmpb(a: vector_float, b: vector_float) -> vector_signed_int {
    sealed::vec_vcmpbfp(a, b)
}

/// Vector ceil.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_ceil(a: vector_float) -> vector_float {
    sealed::vec_vceil(a)
}

/// Vector avg.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_avg<T, U>(a: T, b: U) -> <T as sealed::VectorAvg<U>>::Result
where
    T: sealed::VectorAvg<U>,
{
    a.vec_avg(b)
}

/// Vector andc.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_andc<T, U>(a: T, b: U) -> <T as sealed::VectorAndc<U>>::Result
where
    T: sealed::VectorAndc<U>,
{
    a.vec_andc(b)
}

/// Vector and.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_and<T, U>(a: T, b: U) -> <T as sealed::VectorAnd<U>>::Result
where
    T: sealed::VectorAnd<U>,
{
    a.vec_and(b)
}

/// Vector or.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_or<T, U>(a: T, b: U) -> <T as sealed::VectorOr<U>>::Result
where
    T: sealed::VectorOr<U>,
{
    a.vec_or(b)
}

/// Vector nor.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_nor<T, U>(a: T, b: U) -> <T as sealed::VectorNor<U>>::Result
where
    T: sealed::VectorNor<U>,
{
    a.vec_nor(b)
}

/// Vector xor.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_xor<T, U>(a: T, b: U) -> <T as sealed::VectorXor<U>>::Result
where
    T: sealed::VectorXor<U>,
{
    a.vec_xor(b)
}

/// Vector adds.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_adds<T, U>(a: T, b: U) -> <T as sealed::VectorAdds<U>>::Result
where
    T: sealed::VectorAdds<U>,
{
    a.vec_adds(b)
}

/// Vector addc.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_addc(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int {
    sealed::vec_vaddcuw(a, b)
}

/// Vector abs.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_abs<T>(a: T) -> T
where
    T: sealed::VectorAbs,
{
    a.vec_abs()
}

/// Vector abss.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_abss<T>(a: T) -> T
where
    T: sealed::VectorAbss,
{
    a.vec_abss()
}

/// Vector Splat
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_splat<T, const IMM: u32>(a: T) -> T
where
    T: sealed::VectorSplat,
{
    a.vec_splat::<IMM>()
}

splat! { vec_splat_u8, u8, u8x16 [vspltisb, "Vector Splat to Unsigned Byte"] }
splat! { vec_splat_i8, i8, i8x16 [vspltisb, "Vector Splat to Signed Byte"] }
splat! { vec_splat_u16, u16, u16x8 [vspltish, "Vector Splat to Unsigned Halfword"] }
splat! { vec_splat_i16, i16, i16x8 [vspltish, "Vector Splat to Signed Halfword"] }
splat! { vec_splat_u32, u32, u32x4 [vspltisw, "Vector Splat to Unsigned Word"] }
splat! { vec_splat_i32, i32, i32x4 [vspltisw, "Vector Splat to Signed Word"] }

/// Vector splats.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_splats<T>(a: T) -> <T as sealed::VectorSplats>::Result
where
    T: sealed::VectorSplats,
{
    a.vec_splats()
}

/// Vector sub.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_sub<T, U>(a: T, b: U) -> <T as sealed::VectorSub<U>>::Result
where
    T: sealed::VectorSub<U>,
{
    a.vec_sub(b)
}

/// Vector subs.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_subs<T, U>(a: T, b: U) -> <T as sealed::VectorSubs<U>>::Result
where
    T: sealed::VectorSubs<U>,
{
    a.vec_subs(b)
}

/// Vector min.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_min<T, U>(a: T, b: U) -> <T as sealed::VectorMin<U>>::Result
where
    T: sealed::VectorMin<U>,
{
    a.vec_min(b)
}

/// Vector max.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_max<T, U>(a: T, b: U) -> <T as sealed::VectorMax<U>>::Result
where
    T: sealed::VectorMax<U>,
{
    a.vec_max(b)
}

/// Move From Vector Status and Control Register.
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(mfvscr))]
pub unsafe fn vec_mfvscr() -> vector_unsigned_short {
    mfvscr()
}

/// Vector add.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_add<T, U>(a: T, b: U) -> <T as sealed::VectorAdd<U>>::Result
where
    T: sealed::VectorAdd<U>,
{
    a.vec_add(b)
}

/// Vector Convert to Floating-Point
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_ctf<const IMM5: i32, T>(a: T) -> vector_float
where
    T: sealed::VectorCtf,
{
    a.vec_ctf::<IMM5>()
}

/// Vector Convert to Signed Integer
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(vctsxs, IMM5 = 1))]
pub unsafe fn vec_cts<const IMM5: i32>(a: vector_float) -> vector_signed_int {
    static_assert_uimm_bits!(IMM5, 5);

    vctsxs(a, IMM5)
}

/// Vector Convert to Signed Integer
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(vctuxs, IMM5 = 1))]
pub unsafe fn vec_ctu<const IMM5: i32>(a: vector_float) -> vector_unsigned_int {
    static_assert_uimm_bits!(IMM5, 5);

    vctuxs(a, IMM5)
}

/// Endian-biased intrinsics
#[cfg(target_endian = "little")]
mod endian {
    use super::*;
    /// Vector permute.
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_perm<T>(a: T, b: T, c: vector_unsigned_char) -> T
    where
        T: sealed::VectorPerm,
    {
        // vperm has big-endian bias
        //
        // Xor the mask and flip the arguments
        let d = transmute(u8x16::new(
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ));
        let c = simd_xor(c, d);

        b.vec_vperm(a, c)
    }

    /// Vector Sum Across Partial (1/2) Saturated
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_sum2s(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        // vsum2sws has big-endian bias
        //
        // swap the even b elements with the odd ones
        let flip = transmute(u8x16::new(
            4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11,
        ));
        let b = vec_perm(b, b, flip);
        let c = vsum2sws(a, b);

        vec_perm(c, c, flip)
    }

    // Even and Odd are swapped in little-endian
    /// Vector Multiply Even
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_mule<T, U>(a: T, b: T) -> U
    where
        T: sealed::VectorMulo<U>,
    {
        a.vec_mulo(b)
    }
    /// Vector Multiply Odd
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_mulo<T, U>(a: T, b: T) -> U
    where
        T: sealed::VectorMule<U>,
    {
        a.vec_mule(b)
    }
}

/// Vector Multiply Add Saturated
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(vmhaddshs))]
pub unsafe fn vec_madds(
    a: vector_signed_short,
    b: vector_signed_short,
    c: vector_signed_short,
) -> vector_signed_short {
    vmhaddshs(a, b, c)
}

/// Vector Multiply Low and Add Unsigned Half Word
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_mladd<T, U>(a: T, b: U, c: U) -> <T as sealed::VectorMladd<U>>::Result
where
    T: sealed::VectorMladd<U>,
{
    a.vec_mladd(b, c)
}

/// Vector Multiply Round and Add Saturated
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr(vmhraddshs))]
pub unsafe fn vec_mradds(
    a: vector_signed_short,
    b: vector_signed_short,
    c: vector_signed_short,
) -> vector_signed_short {
    vmhraddshs(a, b, c)
}

/// Vector Multiply Sum
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_msum<T, B, U>(a: T, b: B, c: U) -> U
where
    T: sealed::VectorMsum<B, U>,
{
    a.vec_msum(b, c)
}

/// Vector Multiply Sum Saturated
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_msums<T, U>(a: T, b: T, c: U) -> U
where
    T: sealed::VectorMsums<U>,
{
    a.vec_msums(b, c)
}

/// Vector Multiply Add
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_madd(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
    vmaddfp(a, b, c)
}

/// Vector Negative Multiply Subtract
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_nmsub(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
    vnmsubfp(a, b, c)
}

/// Vector Sum Across Partial (1/4) Saturated
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_sum4s<T, U>(a: T, b: U) -> U
where
    T: sealed::VectorSum4s<U>,
{
    a.vec_sum4s(b)
}

/// Vector All Elements Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_eq<T, U>(a: T, b: U) -> <T as sealed::VectorAllEq<U>>::Result
where
    T: sealed::VectorAllEq<U>,
{
    a.vec_all_eq(b)
}

/// Vector All Elements Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_eq<T, U>(a: T, b: U) -> <T as sealed::VectorAnyEq<U>>::Result
where
    T: sealed::VectorAnyEq<U>,
{
    a.vec_any_eq(b)
}

/// Vector All Elements Greater or Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_ge<T, U>(a: T, b: U) -> <T as sealed::VectorAllGe<U>>::Result
where
    T: sealed::VectorAllGe<U>,
{
    a.vec_all_ge(b)
}

/// Vector Any Element Greater or Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_ge<T, U>(a: T, b: U) -> <T as sealed::VectorAnyGe<U>>::Result
where
    T: sealed::VectorAnyGe<U>,
{
    a.vec_any_ge(b)
}

/// Vector All Elements Greater Than
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_gt<T, U>(a: T, b: U) -> <T as sealed::VectorAllGt<U>>::Result
where
    T: sealed::VectorAllGt<U>,
{
    a.vec_all_gt(b)
}

/// Vector Any Element Greater Than
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_gt<T, U>(a: T, b: U) -> <T as sealed::VectorAnyGt<U>>::Result
where
    T: sealed::VectorAnyGt<U>,
{
    a.vec_any_gt(b)
}

/// Vector All In
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpbfp."))]
pub unsafe fn vec_all_in(a: vector_float, b: vector_float) -> bool {
    vcmpbfp_p(0, a, b) != 0
}

/// Vector All Elements Less Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_le<T, U>(a: U, b: T) -> <T as sealed::VectorAllGe<U>>::Result
where
    T: sealed::VectorAllGe<U>,
{
    b.vec_all_ge(a)
}

/// Vector Any Element Less Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_le<T, U>(a: U, b: T) -> <T as sealed::VectorAnyGe<U>>::Result
where
    T: sealed::VectorAnyGe<U>,
{
    b.vec_any_ge(a)
}

/// Vector All Elements Less Than
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_lt<T, U>(a: U, b: T) -> <T as sealed::VectorAllGt<U>>::Result
where
    T: sealed::VectorAllGt<U>,
{
    b.vec_all_gt(a)
}

/// Vector Any Element Less Than
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_lt<T, U>(a: U, b: T) -> <T as sealed::VectorAnyGt<U>>::Result
where
    T: sealed::VectorAnyGt<U>,
{
    b.vec_any_gt(a)
}

/// All Elements Not a Number
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpeqfp."))]
pub unsafe fn vec_all_nan(a: vector_float) -> bool {
    vcmpeqfp_p(0, a, a) != 0
}

/// Any Elements Not a Number
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpeqfp."))]
pub unsafe fn vec_any_nan(a: vector_float) -> bool {
    vcmpeqfp_p(3, a, a) != 0
}

/// Vector All Elements Not Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_all_ne<T, U>(a: T, b: U) -> <T as sealed::VectorAllNe<U>>::Result
where
    T: sealed::VectorAllNe<U>,
{
    a.vec_all_ne(b)
}

/// Vector Any Elements Not Equal
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_any_ne<T, U>(a: T, b: U) -> <T as sealed::VectorAnyNe<U>>::Result
where
    T: sealed::VectorAnyNe<U>,
{
    a.vec_any_ne(b)
}

/// All Elements Not Greater Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_all_nge(a: vector_float, b: vector_float) -> bool {
    vcmpgefp_p(0, a, b) != 0
}

/// All Elements Not Greater Than
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgtfp."))]
pub unsafe fn vec_all_ngt(a: vector_float, b: vector_float) -> bool {
    vcmpgtfp_p(0, a, b) != 0
}

/// All Elements Not Less Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_all_nle(a: vector_float, b: vector_float) -> bool {
    vcmpgefp_p(0, b, a) != 0
}

/// All Elements Not Less Than
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgtfp."))]
pub unsafe fn vec_all_nlt(a: vector_float, b: vector_float) -> bool {
    vcmpgtfp_p(0, b, a) != 0
}

/// All Elements Numeric
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_all_numeric(a: vector_float) -> bool {
    vcmpgefp_p(2, a, a) != 0
}

/// Any Elements Not Greater Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_any_nge(a: vector_float, b: vector_float) -> bool {
    vcmpgefp_p(3, a, b) != 0
}

/// Any Elements Not Greater Than
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgtfp."))]
pub unsafe fn vec_any_ngt(a: vector_float, b: vector_float) -> bool {
    vcmpgtfp_p(3, a, b) != 0
}

/// Any Elements Not Less Than or Equal
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_any_nle(a: vector_float, b: vector_float) -> bool {
    vcmpgefp_p(3, b, a) != 0
}

/// Any Elements Not Less Than
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgtfp."))]
pub unsafe fn vec_any_nlt(a: vector_float, b: vector_float) -> bool {
    vcmpgtfp_p(3, b, a) != 0
}

/// Any Elements Numeric
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpgefp."))]
pub unsafe fn vec_any_numeric(a: vector_float) -> bool {
    vcmpgefp_p(1, a, a) != 0
}

/// Any Element Out of Bounds
#[inline]
#[target_feature(enable = "altivec")]
#[cfg_attr(test, assert_instr("vcmpeqfp."))]
pub unsafe fn vec_any_out(a: vector_float) -> bool {
    vcmpeqfp_p(1, a, a) != 0
}

#[cfg(target_endian = "big")]
mod endian {
    use super::*;
    /// Vector permute.
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_perm<T>(a: T, b: T, c: vector_unsigned_char) -> T
    where
        T: sealed::VectorPerm,
    {
        a.vec_vperm(b, c)
    }

    /// Vector Sum Across Partial (1/2) Saturated
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_sum2s(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        vsum2sws(a, b)
    }

    /// Vector Multiply Even
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_mule<T, U>(a: T, b: T) -> U
    where
        T: sealed::VectorMule<U>,
    {
        a.vec_mule(b)
    }
    /// Vector Multiply Odd
    #[inline]
    #[target_feature(enable = "altivec")]
    pub unsafe fn vec_mulo<T, U>(a: T, b: T) -> U
    where
        T: sealed::VectorMulo<U>,
    {
        a.vec_mulo(b)
    }
}

pub use self::endian::*;

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "powerpc")]
    use crate::core_arch::arch::powerpc::*;

    #[cfg(target_arch = "powerpc64")]
    use crate::core_arch::arch::powerpc64::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    macro_rules! test_vec_2 {
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! { $name, $fn, $ty -> $ty, [$($a),+], [$($b),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a, b));
                assert_eq!(d, r);
            }
         };
         { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], $d:expr } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let r : $ty_out = transmute($fn(a, b));
                assert_eq!($d, r);
            }
         }
   }

    macro_rules! test_vec_1 {
        { $name: ident, $fn:ident, f32x4, [$($a:expr),+], ~[$($d:expr),+] } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: vector_float = transmute(f32x4::new($($a),+));

                let d: vector_float = transmute(f32x4::new($($d),+));
                let r = transmute(vec_cmple(vec_abs(vec_sub($fn(a), d)), vec_splats(std::f32::EPSILON)));
                let e = m32x4::new(true, true, true, true);
                assert_eq!(e, r);
            }
        };
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($d:expr),+] } => {
            test_vec_1! { $name, $fn, $ty -> $ty, [$($a),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a));
                assert_eq!(d, r);
            }
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_ld() {
        let pat = [
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            u8x16::new(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ),
        ];

        for off in 0..16 {
            let v: u8x16 = transmute(vec_ld(0, (pat.as_ptr() as *const u8).offset(off)));
            assert_eq!(
                v,
                u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            );
        }
        for off in 16..32 {
            let v: u8x16 = transmute(vec_ld(0, (pat.as_ptr() as *const u8).offset(off)));
            assert_eq!(
                v,
                u8x16::new(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
            );
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_xl() {
        let pat = [
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            u8x16::new(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ),
        ];

        for off in 0..16 {
            let val: u8x16 = transmute(vec_xl(0, (pat.as_ptr() as *const u8).offset(off)));
            for i in 0..16 {
                let v = val.extract(i);
                assert_eq!(off as usize + i, v as usize);
            }
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_ldl() {
        let pat = [
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            u8x16::new(
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ),
        ];

        for off in 0..16 {
            let v: u8x16 = transmute(vec_ldl(0, (pat.as_ptr() as *const u8).offset(off)));
            assert_eq!(
                v,
                u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            );
        }
        for off in 16..32 {
            let v: u8x16 = transmute(vec_ldl(0, (pat.as_ptr() as *const u8).offset(off)));
            assert_eq!(
                v,
                u8x16::new(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
            );
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_lde_u8() {
        let pat = [u8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        )];
        for off in 0..16 {
            let v: u8x16 = transmute(vec_lde(off, pat.as_ptr() as *const u8));
            assert_eq!(off as u8, v.extract(off as _));
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_lde_u16() {
        let pat = [u16x8::new(0, 1, 2, 3, 4, 5, 6, 7)];
        for off in 0..8 {
            let v: u16x8 = transmute(vec_lde(off * 2, pat.as_ptr() as *const u8));
            assert_eq!(off as u16, v.extract(off as _));
        }
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_lde_u32() {
        let pat = [u32x4::new(0, 1, 2, 3)];
        for off in 0..4 {
            let v: u32x4 = transmute(vec_lde(off * 4, pat.as_ptr() as *const u8));
            assert_eq!(off as u32, v.extract(off as _));
        }
    }

    test_vec_1! { test_vec_floor, vec_floor, f32x4,
        [1.1, 1.9, -0.5, -0.9],
        [1.0, 1.0, -1.0, -1.0]
    }

    test_vec_1! { test_vec_expte, vec_expte, f32x4,
        [0.0, 2.0, 2.0, -1.0],
        ~[1.0, 4.0, 4.0, 0.5]
    }

    test_vec_2! { test_vec_cmpgt_i8, vec_cmpgt, i8x16 -> m8x16,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [true, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false]
    }

    test_vec_2! { test_vec_cmpgt_u8, vec_cmpgt, u8x16 -> m8x16,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false]
    }

    test_vec_2! { test_vec_cmpgt_i16, vec_cmpgt, i16x8 -> m16x8,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [true, false, true, false, false, false, false, false]
    }

    test_vec_2! { test_vec_cmpgt_u16, vec_cmpgt, u16x8 -> m16x8,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0],
        [true, true, false, false, false, false, false, false]
    }

    test_vec_2! { test_vec_cmpgt_i32, vec_cmpgt, i32x4 -> m32x4,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        [true, false, false, false]
    }

    test_vec_2! { test_vec_cmpgt_u32, vec_cmpgt, u32x4 -> m32x4,
        [1, 255, 0, 0],
        [0, 255,  0, 1],
        [true, false, false, false]
    }

    test_vec_2! { test_vec_cmpge, vec_cmpge, f32x4 -> m32x4,
        [0.1, -0.1, 0.0, 0.99],
        [0.1, 0.0, 0.1, 1.0],
        [true, false, false, false]
    }

    test_vec_2! { test_vec_cmpeq_i8, vec_cmpeq, i8x16 -> m8x16,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true]
    }

    test_vec_2! { test_vec_cmpeq_u8, vec_cmpeq, u8x16 -> m8x16,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true]
    }

    test_vec_2! { test_vec_cmpeq_i16, vec_cmpeq, i16x8 -> m16x8,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [false, false, false, false, true, true, true, true]
    }

    test_vec_2! { test_vec_cmpeq_u16, vec_cmpeq, u16x8 -> m16x8,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0],
        [false, false, false, false, true, true, true, true]
    }

    test_vec_2! { test_vec_cmpeq_i32, vec_cmpeq, i32x4 -> m32x4,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        [false, true, true, false]
    }

    test_vec_2! { test_vec_cmpeq_u32, vec_cmpeq, u32x4 -> m32x4,
        [1, 255, 0, 0],
        [0, 255,  0, 1],
        [false, true, true, false]
    }

    test_vec_2! { test_vec_all_eq_i8_false, vec_all_eq, i8x16 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_eq_u8_false, vec_all_eq, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_eq_i16_false, vec_all_eq, i16x8 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_eq_u16_false, vec_all_eq, u16x8 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_eq_i32_false, vec_all_eq, i32x4 -> bool,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        false
    }

    test_vec_2! { test_vec_all_eq_u32_false, vec_all_eq, u32x4 -> bool,
        [1, 255, 0, 0],
        [0, 255,  0, 1],
        false
    }

    test_vec_2! { test_vec_all_eq_i8_true, vec_all_eq, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_eq_u8_true, vec_all_eq, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_eq_i16_true, vec_all_eq, i16x8 -> bool,
        [1, -1, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_eq_u16_true, vec_all_eq, u16x8 -> bool,
        [1, 255, 1, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_eq_i32_true, vec_all_eq, i32x4 -> bool,
        [1, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_eq_u32_true, vec_all_eq, u32x4 -> bool,
        [1, 255, 0, 1],
        [1, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_eq_i8_false, vec_any_eq, i8x16 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_u8_false, vec_any_eq, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_i16_false, vec_any_eq, i16x8 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_u16_false, vec_any_eq, u16x8 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_i32_false, vec_any_eq, i32x4 -> bool,
        [1, -1, 0, 0],
        [0, -2, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_u32_false, vec_any_eq, u32x4 -> bool,
        [1, 2, 1, 0],
        [0, 255,  0, 1],
        false
    }

    test_vec_2! { test_vec_any_eq_i8_true, vec_any_eq, i8x16 -> bool,
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_eq_u8_true, vec_any_eq, u8x16 -> bool,
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_eq_i16_true, vec_any_eq, i16x8 -> bool,
        [0, -1, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_eq_u16_true, vec_any_eq, u16x8 -> bool,
        [0, 255, 1, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_eq_i32_true, vec_any_eq, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_eq_u32_true, vec_any_eq, u32x4 -> bool,
        [0, 255, 0, 1],
        [1, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_ge_i8_false, vec_all_ge, i8x16 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ge_u8_false, vec_all_ge, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ge_i16_false, vec_all_ge, i16x8 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ge_u16_false, vec_all_ge, u16x8 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ge_i32_false, vec_all_ge, i32x4 -> bool,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        false
    }

    test_vec_2! { test_vec_all_ge_u32_false, vec_all_ge, u32x4 -> bool,
        [1, 255, 0, 0],
        [0, 255,  1, 1],
        false
    }

    test_vec_2! { test_vec_all_ge_i8_true, vec_all_ge, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ge_u8_true, vec_all_ge, u8x16 -> bool,
        [1, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ge_i16_true, vec_all_ge, i16x8 -> bool,
        [1, -1, 42, 0, 0, 0, 0, 0],
        [1, -5, 2, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ge_u16_true, vec_all_ge, u16x8 -> bool,
        [42, 255, 1, 0, 0, 0, 0, 0],
        [2, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ge_i32_true, vec_all_ge, i32x4 -> bool,
        [1, -1, 0, 1],
        [0, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_ge_u32_true, vec_all_ge, u32x4 -> bool,
        [1, 255, 0, 1],
        [1, 254, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ge_i8_false, vec_any_ge, i8x16 -> bool,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_u8_false, vec_any_ge, u8x16 -> bool,
        [1, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [42, 255, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_i16_false, vec_any_ge, i16x8 -> bool,
        [1, -1, -2, 0, 0, 0, 0, 0],
        [2, 0, -1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_u16_false, vec_any_ge, u16x8 -> bool,
        [1, 2, 0, 0, 0, 0, 0, 0],
        [2, 42, 255, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_i32_false, vec_any_ge, i32x4 -> bool,
        [1, -1, 0, 0],
        [2, 0, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_u32_false, vec_any_ge, u32x4 -> bool,
        [1, 2, 1, 0],
        [4, 255,  4, 1],
        false
    }

    test_vec_2! { test_vec_any_ge_i8_true, vec_any_ge, i8x16 -> bool,
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ge_u8_true, vec_any_ge, u8x16 -> bool,
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ge_i16_true, vec_any_ge, i16x8 -> bool,
        [0, -1, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ge_u16_true, vec_any_ge, u16x8 -> bool,
        [0, 255, 1, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ge_i32_true, vec_any_ge, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_ge_u32_true, vec_any_ge, u32x4 -> bool,
        [0, 255, 0, 1],
        [1, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_gt_i8_false, vec_all_gt, i8x16 -> bool,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_gt_u8_false, vec_all_gt, u8x16 -> bool,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_gt_i16_false, vec_all_gt, i16x8 -> bool,
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_gt_u16_false, vec_all_gt, u16x8 -> bool,
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_gt_i32_false, vec_all_gt, i32x4 -> bool,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        false
    }

    test_vec_2! { test_vec_all_gt_u32_false, vec_all_gt, u32x4 -> bool,
        [1, 255, 0, 0],
        [0, 255,  1, 1],
        false
    }

    test_vec_2! { test_vec_all_gt_i8_true, vec_all_gt, i8x16 -> bool,
        [2, 1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        true
    }

    test_vec_2! { test_vec_all_gt_u8_true, vec_all_gt, u8x16 -> bool,
        [1, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_gt_i16_true, vec_all_gt, i16x8 -> bool,
        [1, -1, 42, 1, 1, 1, 1, 1],
        [0, -5, 2, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_gt_u16_true, vec_all_gt, u16x8 -> bool,
        [42, 255, 1, 1, 1, 1, 1, 1],
        [2, 254, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_gt_i32_true, vec_all_gt, i32x4 -> bool,
        [1, -1, 1, 1],
        [0, -2, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_gt_u32_true, vec_all_gt, u32x4 -> bool,
        [1, 255, 1, 1],
        [0, 254, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_gt_i8_false, vec_any_gt, i8x16 -> bool,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_u8_false, vec_any_gt, u8x16 -> bool,
        [1, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [42, 255, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_i16_false, vec_any_gt, i16x8 -> bool,
        [1, -1, -2, 0, 0, 0, 0, 0],
        [2, 0, -1, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_u16_false, vec_any_gt, u16x8 -> bool,
        [1, 2, 0, 0, 0, 0, 0, 0],
        [2, 42, 255, 1, 1, 1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_i32_false, vec_any_gt, i32x4 -> bool,
        [1, -1, 0, 0],
        [2, 0, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_u32_false, vec_any_gt, u32x4 -> bool,
        [1, 2, 1, 0],
        [4, 255,  4, 1],
        false
    }

    test_vec_2! { test_vec_any_gt_i8_true, vec_any_gt, i8x16 -> bool,
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_gt_u8_true, vec_any_gt, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_gt_i16_true, vec_any_gt, i16x8 -> bool,
        [1, -1, 1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_gt_u16_true, vec_any_gt, u16x8 -> bool,
        [1, 255, 1, 0, 0, 0, 0, 0],
        [0, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_gt_i32_true, vec_any_gt, i32x4 -> bool,
        [1, -1, 0, 1],
        [0, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_gt_u32_true, vec_any_gt, u32x4 -> bool,
        [1, 255, 0, 1],
        [0, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_in_true, vec_all_in, f32x4 -> bool,
        [0.0, -0.1, 0.0, 0.0],
        [0.1, 0.2, 0.0, 0.0],
        true
    }

    test_vec_2! { test_vec_all_in_false, vec_all_in, f32x4 -> bool,
        [0.5, 0.4, -0.5, 0.8],
        [0.1, 0.4, -0.5, 0.8],
        false
    }

    test_vec_2! { test_vec_all_le_i8_false, vec_all_le, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_u8_false, vec_all_le, u8x16 -> bool,
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_i16_false, vec_all_le, i16x8 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_u16_false, vec_all_le, u16x8 -> bool,
        [0, 0, 255, 1, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_i32_false, vec_all_le, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_u32_false, vec_all_le, u32x4 -> bool,
        [0, 255,  1, 1],
        [1, 255, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_le_i8_true, vec_all_le, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_le_u8_true, vec_all_le, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_le_i16_true, vec_all_le, i16x8 -> bool,
        [1, -5, 2, 0, 0, 0, 0, 0],
        [1, -1, 42, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_le_u16_true, vec_all_le, u16x8 -> bool,
        [2, 255, 1, 0, 0, 0, 0, 0],
        [42, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_le_i32_true, vec_all_le, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_le_u32_true, vec_all_le, u32x4 -> bool,
        [1, 254, 0, 0],
        [1, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_le_i8_false, vec_any_le, i8x16 -> bool,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_le_u8_false, vec_any_le, u8x16 -> bool,
        [42, 255, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_le_i16_false, vec_any_le, i16x8 -> bool,
        [2, 0, -1, 1, 1, 1, 1, 1],
        [1, -1, -2, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_le_u16_false, vec_any_le, u16x8 -> bool,
        [2, 42, 255, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_le_i32_false, vec_any_le, i32x4 -> bool,
        [2, 0, 1, 1],
        [1, -1, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_le_u32_false, vec_any_le, u32x4 -> bool,
        [4, 255,  4, 1],
        [1, 2, 1, 0],
        false
    }

    test_vec_2! { test_vec_any_le_i8_true, vec_any_le, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_le_u8_true, vec_any_le, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_le_i16_true, vec_any_le, i16x8 -> bool,
        [1, -1, 1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_le_u16_true, vec_any_le, u16x8 -> bool,
        [1, 255, 1, 0, 0, 0, 0, 0],
        [0, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_le_i32_true, vec_any_le, i32x4 -> bool,
        [1, -1, 0, 1],
        [0, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_le_u32_true, vec_any_le, u32x4 -> bool,
        [1, 255, 0, 1],
        [0, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_lt_i8_false, vec_all_lt, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_u8_false, vec_all_lt, u8x16 -> bool,
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_i16_false, vec_all_lt, i16x8 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_u16_false, vec_all_lt, u16x8 -> bool,
        [0, 0, 255, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_i32_false, vec_all_lt, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_u32_false, vec_all_lt, u32x4 -> bool,
        [0, 255,  1, 1],
        [1, 255, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_lt_i8_true, vec_all_lt, i8x16 -> bool,
        [0, 0, -2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_lt_u8_true, vec_all_lt, u8x16 -> bool,
        [0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        true
    }

    test_vec_2! { test_vec_all_lt_i16_true, vec_all_lt, i16x8 -> bool,
        [0, -5, 2, 0, 0, 0, 0, 0],
        [1, -1, 42, 1, 1, 1, 1, 1],
        true
    }

    test_vec_2! { test_vec_all_lt_u16_true, vec_all_lt, u16x8 -> bool,
        [2, 254, 0, 0, 0, 0, 0, 0],
        [42, 255, 1, 1, 1, 1, 1, 1],
        true
    }

    test_vec_2! { test_vec_all_lt_i32_true, vec_all_lt, i32x4 -> bool,
        [0, -2, 0, 0],
        [1, -1, 1, 1],
        true
    }

    test_vec_2! { test_vec_all_lt_u32_true, vec_all_lt, u32x4 -> bool,
        [0, 254, 0, 0],
        [1, 255, 1, 1],
        true
    }

    test_vec_2! { test_vec_any_lt_i8_false, vec_any_lt, i8x16 -> bool,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_u8_false, vec_any_lt, u8x16 -> bool,
        [42, 255, 255, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_i16_false, vec_any_lt, i16x8 -> bool,
        [2, 0, -1, 1, 1, 1, 1, 1],
        [1, -1, -2, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_u16_false, vec_any_lt, u16x8 -> bool,
        [2, 42, 255, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_i32_false, vec_any_lt, i32x4 -> bool,
        [2, 0, 1, 1],
        [1, -1, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_u32_false, vec_any_lt, u32x4 -> bool,
        [4, 255,  4, 1],
        [1, 2, 1, 0],
        false
    }

    test_vec_2! { test_vec_any_lt_i8_true, vec_any_lt, i8x16 -> bool,
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_lt_u8_true, vec_any_lt, u8x16 -> bool,
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_lt_i16_true, vec_any_lt, i16x8 -> bool,
        [0, -1, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_lt_u16_true, vec_any_lt, u16x8 -> bool,
        [0, 255, 1, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_lt_i32_true, vec_any_lt, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_lt_u32_true, vec_any_lt, u32x4 -> bool,
        [0, 255, 0, 1],
        [1, 255, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_ne_i8_false, vec_all_ne, i8x16 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ne_u8_false, vec_all_ne, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ne_i16_false, vec_all_ne, i16x8 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ne_u16_false, vec_all_ne, u16x8 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0],
        [0, 255, 0, 1, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_all_ne_i32_false, vec_all_ne, i32x4 -> bool,
        [1, -1, 0, 0],
        [0, -1, 0, 1],
        false
    }

    test_vec_2! { test_vec_all_ne_u32_false, vec_all_ne, u32x4 -> bool,
        [1, 255, 0, 0],
        [0, 255,  0, 1],
        false
    }

    test_vec_2! { test_vec_all_ne_i8_true, vec_all_ne, i8x16 -> bool,
        [0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ne_u8_true, vec_all_ne, u8x16 -> bool,
        [0, 254, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ne_i16_true, vec_all_ne, i16x8 -> bool,
        [2, -2, 0, 1, 1, 1, 1, 1],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_all_ne_u16_true, vec_all_ne, u16x8 -> bool,
        [0, 254, 1, 1, 0, 0, 1, 0],
        [1, 255, 0, 0, 1, 1, 0, 1],
        true
    }

    test_vec_2! { test_vec_all_ne_i32_true, vec_all_ne, i32x4 -> bool,
        [0, -2, 0, 0],
        [1, -1, 1, 1],
        true
    }

    test_vec_2! { test_vec_all_ne_u32_true, vec_all_ne, u32x4 -> bool,
        [1, 255, 0, 0],
        [0, 254, 1, 1],
        true
    }

    test_vec_2! { test_vec_any_ne_i8_false, vec_any_ne, i8x16 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_ne_u8_false, vec_any_ne, u8x16 -> bool,
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_ne_i16_false, vec_any_ne, i16x8 -> bool,
        [1, -1, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        false
    }

    test_vec_2! { test_vec_any_ne_u16_false, vec_any_ne, u16x8 -> bool,
        [1, 255, 1, 1, 1, 1, 1, 0],
        [1, 255, 1, 1, 1, 1, 1, 0],
        false
    }

    test_vec_2! { test_vec_any_ne_i32_false, vec_any_ne, i32x4 -> bool,
        [0, -1, 1, 1],
        [0, -1, 1, 1],
        false
    }

    test_vec_2! { test_vec_any_ne_u32_false, vec_any_ne, u32x4 -> bool,
        [1, 2, 1, 255],
        [1, 2, 1, 255],
        false
    }

    test_vec_2! { test_vec_any_ne_i8_true, vec_any_ne, i8x16 -> bool,
        [1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ne_u8_true, vec_any_ne, u8x16 -> bool,
        [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ne_i16_true, vec_any_ne, i16x8 -> bool,
        [0, -1, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ne_u16_true, vec_any_ne, u16x8 -> bool,
        [0, 255, 1, 0, 0, 0, 0, 0],
        [1, 255, 1, 0, 0, 0, 0, 0],
        true
    }

    test_vec_2! { test_vec_any_ne_i32_true, vec_any_ne, i32x4 -> bool,
        [0, -1, 0, 1],
        [1, -1, 0, 1],
        true
    }

    test_vec_2! { test_vec_any_ne_u32_true, vec_any_ne, u32x4 -> bool,
        [0, 255, 0, 1],
        [1, 255, 0, 1],
        true
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_cmpb() {
        let a: vector_float = transmute(f32x4::new(0.1, 0.5, 0.6, 0.9));
        let b: vector_float = transmute(f32x4::new(-0.1, 0.5, -0.6, 0.9));
        let d = i32x4::new(
            -0b10000000000000000000000000000000,
            0,
            -0b10000000000000000000000000000000,
            0,
        );

        assert_eq!(d, transmute(vec_cmpb(a, b)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_ceil() {
        let a: vector_float = transmute(f32x4::new(0.1, 0.5, 0.6, 0.9));
        let d = f32x4::new(1.0, 1.0, 1.0, 1.0);

        assert_eq!(d, transmute(vec_ceil(a)));
    }

    test_vec_2! { test_vec_andc, vec_andc, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b10000000],
    [0b11001100, 0b00001100, 0b11000000, 0b01001100] }

    test_vec_2! { test_vec_and, vec_and, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b00000000, 0b11000000, 0b00001100, 0b00000000] }

    macro_rules! test_vec_avg {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_avg, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_avg! { test_vec_avg_i32x4, i32x4,
    [i32::MIN, i32::MAX, 1, -1],
    [-1, 1, 1, -1],
    [-1073741824, 1073741824, 1, -1] }

    test_vec_avg! { test_vec_avg_u32x4, u32x4,
    [u32::MAX, 0, 1, 2],
    [2, 1, 0, 0],
    [2147483649, 1, 1, 1] }

    test_vec_avg! { test_vec_avg_i16x8, i16x8,
    [i16::MIN, i16::MAX, 1, -1, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0],
    [-16384, 16384, 1, -1, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_u16x8, u16x8,
    [u16::MAX, 0, 1, 2, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0],
    [32769, 1, 1, 1, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_i8x16, i8x16,
    [i8::MIN, i8::MAX, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-64, 64, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_u8x16, u8x16,
    [u8::MAX, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [129, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    macro_rules! test_vec_adds {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_adds, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_adds! { test_vec_adds_i32x4, i32x4,
    [i32::MIN, i32::MAX, 1, -1],
    [-1, 1, 1, -1],
    [i32::MIN, i32::MAX, 2, -2] }

    test_vec_adds! { test_vec_adds_u32x4, u32x4,
    [u32::MAX, 0, 1, 2],
    [2, 1, 0, 0],
    [u32::MAX, 1, 1, 2] }

    test_vec_adds! { test_vec_adds_i16x8, i16x8,
    [i16::MIN, i16::MAX, 1, -1, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0],
    [i16::MIN, i16::MAX, 2, -2, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_u16x8, u16x8,
    [u16::MAX, 0, 1, 2, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0],
    [u16::MAX, 1, 1, 2, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_i8x16, i8x16,
    [i8::MIN, i8::MAX, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [i8::MIN, i8::MAX, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_u8x16, u8x16,
    [u8::MAX, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [u8::MAX, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_2! { test_vec_addc, vec_addc, u32x4, [u32::MAX, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0] }

    macro_rules! test_vec_abs {
        { $name: ident, $ty: ident, $a: expr, $d: expr } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a = vec_splats($a);
                let a: s_t_l!($ty) = vec_abs(a);
                let d = $ty::splat($d);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_abs! { test_vec_abs_i8, i8x16, -42i8, 42i8 }
    test_vec_abs! { test_vec_abs_i16, i16x8, -42i16, 42i16 }
    test_vec_abs! { test_vec_abs_i32, i32x4, -42i32, 42i32 }
    test_vec_abs! { test_vec_abs_f32, f32x4, -42f32, 42f32 }

    macro_rules! test_vec_abss {
        { $name: ident, $ty: ident, $a: expr, $d: expr } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a = vec_splats($a);
                let a: s_t_l!($ty) = vec_abss(a);
                let d = $ty::splat($d);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_abss! { test_vec_abss_i8, i8x16, -127i8, 127i8 }
    test_vec_abss! { test_vec_abss_i16, i16x8, -42i16, 42i16 }
    test_vec_abss! { test_vec_abss_i32, i32x4, -42i32, 42i32 }

    macro_rules! test_vec_splats {
        { $name: ident, $ty: ident, $a: expr } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = vec_splats($a);
                let d = $ty::splat($a);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_splats! { test_vec_splats_u8, u8x16, 42u8 }
    test_vec_splats! { test_vec_splats_u16, u16x8, 42u16 }
    test_vec_splats! { test_vec_splats_u32, u32x4, 42u32 }
    test_vec_splats! { test_vec_splats_i8, i8x16, 42i8 }
    test_vec_splats! { test_vec_splats_i16, i16x8, 42i16 }
    test_vec_splats! { test_vec_splats_i32, i32x4, 42i32 }
    test_vec_splats! { test_vec_splats_f32, f32x4, 42f32 }

    macro_rules! test_vec_splat {
        { $name: ident, $fun: ident, $ty: ident, $a: expr, $b: expr} => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a = $fun::<$a>();
                let d = $ty::splat($b);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_splat! { test_vec_splat_u8, vec_splat_u8, u8x16, -1, u8::MAX }
    test_vec_splat! { test_vec_splat_u16, vec_splat_u16, u16x8, -1, u16::MAX }
    test_vec_splat! { test_vec_splat_u32, vec_splat_u32, u32x4, -1, u32::MAX }
    test_vec_splat! { test_vec_splat_i8, vec_splat_i8, i8x16, -1, -1 }
    test_vec_splat! { test_vec_splat_i16, vec_splat_i16, i16x8, -1, -1 }
    test_vec_splat! { test_vec_splat_i32, vec_splat_i32, i32x4, -1, -1 }

    macro_rules! test_vec_sub {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_sub, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_sub! { test_vec_sub_f32x4, f32x4,
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, -1.0, -2.0],
    [-3.0, -1.0, 2.0, 4.0] }

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

    macro_rules! test_vec_subs {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_subs, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_subs! { test_vec_subs_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-3, -1, 2, 4] }

    test_vec_subs! { test_vec_subs_u32x4, u32x4,
    [0, 0, 1, 2],
    [2, 1, 0, 0],
    [0, 0, 1, 2] }

    test_vec_subs! { test_vec_subs_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_subs! { test_vec_subs_u16x8, u16x8,
    [0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2] }

    test_vec_subs! { test_vec_subs_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_subs! { test_vec_subs_u8x16, u8x16,
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2] }

    macro_rules! test_vec_min {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let d = $ty::new($($d),+);
                let r : $ty = transmute(vec_min(a, b));
                assert_eq!(d, r);
            }
         }
    }

    test_vec_min! { test_vec_min_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-1, 0, -1, -2] }

    test_vec_min! { test_vec_min_u32x4, u32x4,
    [0, 0, 1, 2],
    [2, 1, 0, 0],
    [0, 0, 0, 0] }

    test_vec_min! { test_vec_min_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-1, 0, -1, -2, -1, 0, -1, -2] }

    test_vec_min! { test_vec_min_u16x8, u16x8,
    [0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_min! { test_vec_min_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2] }

    test_vec_min! { test_vec_min_u8x16, u8x16,
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    macro_rules! test_vec_max {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let d = $ty::new($($d),+);
                let r : $ty = transmute(vec_max(a, b));
                assert_eq!(d, r);
            }
         }
    }

    test_vec_max! { test_vec_max_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [2, 1, 1, 2] }

    test_vec_max! { test_vec_max_u32x4, u32x4,
    [0, 0, 1, 2],
    [2, 1, 0, 0],
    [2, 1, 1, 2] }

    test_vec_max! { test_vec_max_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [2, 1, 1, 2, 2, 1, 1, 2] }

    test_vec_max! { test_vec_max_u16x8, u16x8,
    [0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0],
    [2, 1, 1, 2, 2, 1, 1, 2] }

    test_vec_max! { test_vec_max_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2] }

    test_vec_max! { test_vec_max_u8x16, u8x16,
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
    [2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2] }

    macro_rules! test_vec_perm {
        {$name:ident,
         $shorttype:ident, $longtype:ident,
         [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "altivec")]
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

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_madds() {
        let a: vector_signed_short = transmute(i16x8::new(
            0 * 256,
            1 * 256,
            2 * 256,
            3 * 256,
            4 * 256,
            5 * 256,
            6 * 256,
            7 * 256,
        ));
        let b: vector_signed_short = transmute(i16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_signed_short = transmute(i16x8::new(0, 1, 2, 3, 4, 5, 6, 7));

        let d = i16x8::new(0, 3, 6, 9, 12, 15, 18, 21);

        assert_eq!(d, transmute(vec_madds(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_madd_float() {
        let a: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let b: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let c: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let d = f32x4::new(
            0.1 * 0.1 + 0.1,
            0.2 * 0.2 + 0.2,
            0.3 * 0.3 + 0.3,
            0.4 * 0.4 + 0.4,
        );

        assert_eq!(d, transmute(vec_madd(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_nmsub_float() {
        let a: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let b: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let c: vector_float = transmute(f32x4::new(0.1, 0.2, 0.3, 0.4));
        let d = f32x4::new(
            -(0.1 * 0.1 - 0.1),
            -(0.2 * 0.2 - 0.2),
            -(0.3 * 0.3 - 0.3),
            -(0.4 * 0.4 - 0.4),
        );
        assert_eq!(d, transmute(vec_nmsub(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mradds() {
        let a: vector_signed_short = transmute(i16x8::new(
            0 * 256,
            1 * 256,
            2 * 256,
            3 * 256,
            4 * 256,
            5 * 256,
            6 * 256,
            7 * 256,
        ));
        let b: vector_signed_short = transmute(i16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_signed_short = transmute(i16x8::new(0, 1, 2, 3, 4, 5, 6, i16::MAX - 1));

        let d = i16x8::new(0, 3, 6, 9, 12, 15, 18, i16::MAX);

        assert_eq!(d, transmute(vec_mradds(a, b, c)));
    }

    macro_rules! test_vec_mladd {
        {$name:ident, $sa:ident, $la:ident, $sbc:ident, $lbc:ident, $sd:ident,
            [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "altivec")]
            unsafe fn $name() {
                let a: $la = transmute($sa::new($($a),+));
                let b: $lbc = transmute($sbc::new($($b),+));
                let c = transmute($sbc::new($($c),+));
                let d = $sd::new($($d),+);

                assert_eq!(d, transmute(vec_mladd(a, b, c)));
            }
        }
    }

    test_vec_mladd! { test_vec_mladd_u16x8_u16x8, u16x8, vector_unsigned_short, u16x8, vector_unsigned_short, u16x8,
        [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 6, 12, 20, 30, 42, 56]
    }
    test_vec_mladd! { test_vec_mladd_u16x8_i16x8, u16x8, vector_unsigned_short, i16x8, vector_unsigned_short, i16x8,
        [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 6, 12, 20, 30, 42, 56]
    }
    test_vec_mladd! { test_vec_mladd_i16x8_u16x8, i16x8, vector_signed_short, u16x8, vector_unsigned_short, i16x8,
        [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 6, 12, 20, 30, 42, 56]
    }
    test_vec_mladd! { test_vec_mladd_i16x8_i16x8, i16x8, vector_signed_short, i16x8, vector_unsigned_short, i16x8,
        [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 6, 12, 20, 30, 42, 56]
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msum_unsigned_char() {
        let a: vector_unsigned_char =
            transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let b: vector_unsigned_char = transmute(u8x16::new(
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ));
        let c: vector_unsigned_int = transmute(u32x4::new(0, 1, 2, 3));
        let d = u32x4::new(
            (0 + 1 + 2 + 3) * 255 + 0,
            (4 + 5 + 6 + 7) * 255 + 1,
            (0 + 1 + 2 + 3) * 255 + 2,
            (4 + 5 + 6 + 7) * 255 + 3,
        );

        assert_eq!(d, transmute(vec_msum(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msum_signed_char() {
        let a: vector_signed_char = transmute(i8x16::new(
            0, -1, 2, -3, 1, -1, 1, -1, 0, 1, 2, 3, 4, -5, -6, -7,
        ));
        let b: vector_unsigned_char =
            transmute(i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
        let c: vector_signed_int = transmute(u32x4::new(0, 1, 2, 3));
        let d = i32x4::new(
            (0 - 1 + 2 - 3) + 0,
            (0) + 1,
            (0 + 1 + 2 + 3) + 2,
            (4 - 5 - 6 - 7) + 3,
        );

        assert_eq!(d, transmute(vec_msum(a, b, c)));
    }
    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msum_unsigned_short() {
        let a: vector_unsigned_short = transmute(u16x8::new(
            0 * 256,
            1 * 256,
            2 * 256,
            3 * 256,
            4 * 256,
            5 * 256,
            6 * 256,
            7 * 256,
        ));
        let b: vector_unsigned_short =
            transmute(u16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_unsigned_int = transmute(u32x4::new(0, 1, 2, 3));
        let d = u32x4::new(
            (0 + 1) * 256 * 256 + 0,
            (2 + 3) * 256 * 256 + 1,
            (4 + 5) * 256 * 256 + 2,
            (6 + 7) * 256 * 256 + 3,
        );

        assert_eq!(d, transmute(vec_msum(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msum_signed_short() {
        let a: vector_signed_short = transmute(i16x8::new(
            0 * 256,
            -1 * 256,
            2 * 256,
            -3 * 256,
            4 * 256,
            -5 * 256,
            6 * 256,
            -7 * 256,
        ));
        let b: vector_signed_short = transmute(i16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let d = i32x4::new(
            (0 - 1) * 256 * 256 + 0,
            (2 - 3) * 256 * 256 + 1,
            (4 - 5) * 256 * 256 + 2,
            (6 - 7) * 256 * 256 + 3,
        );

        assert_eq!(d, transmute(vec_msum(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msums_unsigned() {
        let a: vector_unsigned_short = transmute(u16x8::new(
            0 * 256,
            1 * 256,
            2 * 256,
            3 * 256,
            4 * 256,
            5 * 256,
            6 * 256,
            7 * 256,
        ));
        let b: vector_unsigned_short =
            transmute(u16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_unsigned_int = transmute(u32x4::new(0, 1, 2, 3));
        let d = u32x4::new(
            (0 + 1) * 256 * 256 + 0,
            (2 + 3) * 256 * 256 + 1,
            (4 + 5) * 256 * 256 + 2,
            (6 + 7) * 256 * 256 + 3,
        );

        assert_eq!(d, transmute(vec_msums(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_msums_signed() {
        let a: vector_signed_short = transmute(i16x8::new(
            0 * 256,
            -1 * 256,
            2 * 256,
            -3 * 256,
            4 * 256,
            -5 * 256,
            6 * 256,
            -7 * 256,
        ));
        let b: vector_signed_short = transmute(i16x8::new(256, 256, 256, 256, 256, 256, 256, 256));
        let c: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let d = i32x4::new(
            (0 - 1) * 256 * 256 + 0,
            (2 - 3) * 256 * 256 + 1,
            (4 - 5) * 256 * 256 + 2,
            (6 - 7) * 256 * 256 + 3,
        );

        assert_eq!(d, transmute(vec_msums(a, b, c)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_sum2s() {
        let a: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let b: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let d = i32x4::new(0, 0 + 1 + 1, 0, 2 + 3 + 3);

        assert_eq!(d, transmute(vec_sum2s(a, b)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_sum4s_unsigned_char() {
        let a: vector_unsigned_char =
            transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let b: vector_unsigned_int = transmute(u32x4::new(0, 1, 2, 3));
        let d = u32x4::new(
            0 + 1 + 2 + 3 + 0,
            4 + 5 + 6 + 7 + 1,
            0 + 1 + 2 + 3 + 2,
            4 + 5 + 6 + 7 + 3,
        );

        assert_eq!(d, transmute(vec_sum4s(a, b)));
    }
    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_sum4s_signed_char() {
        let a: vector_signed_char =
            transmute(i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let b: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let d = i32x4::new(
            0 + 1 + 2 + 3 + 0,
            4 + 5 + 6 + 7 + 1,
            0 + 1 + 2 + 3 + 2,
            4 + 5 + 6 + 7 + 3,
        );

        assert_eq!(d, transmute(vec_sum4s(a, b)));
    }
    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_sum4s_signed_short() {
        let a: vector_signed_short = transmute(i16x8::new(0, 1, 2, 3, 4, 5, 6, 7));
        let b: vector_signed_int = transmute(i32x4::new(0, 1, 2, 3));
        let d = i32x4::new(0 + 1 + 0, 2 + 3 + 1, 4 + 5 + 2, 6 + 7 + 3);

        assert_eq!(d, transmute(vec_sum4s(a, b)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mule_unsigned_char() {
        let a: vector_unsigned_char =
            transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let d = u16x8::new(0 * 0, 2 * 2, 4 * 4, 6 * 6, 0 * 0, 2 * 2, 4 * 4, 6 * 6);

        assert_eq!(d, transmute(vec_mule(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mule_signed_char() {
        let a: vector_signed_char = transmute(i8x16::new(
            0, 1, -2, 3, -4, 5, -6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
        ));
        let d = i16x8::new(0 * 0, 2 * 2, 4 * 4, 6 * 6, 0 * 0, 2 * 2, 4 * 4, 6 * 6);

        assert_eq!(d, transmute(vec_mule(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mule_unsigned_short() {
        let a: vector_unsigned_short = transmute(u16x8::new(0, 1, 2, 3, 4, 5, 6, 7));
        let d = u32x4::new(0 * 0, 2 * 2, 4 * 4, 6 * 6);

        assert_eq!(d, transmute(vec_mule(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mule_signed_short() {
        let a: vector_signed_short = transmute(i16x8::new(0, 1, -2, 3, -4, 5, -6, 7));
        let d = i32x4::new(0 * 0, 2 * 2, 4 * 4, 6 * 6);

        assert_eq!(d, transmute(vec_mule(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mulo_unsigned_char() {
        let a: vector_unsigned_char =
            transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let d = u16x8::new(1 * 1, 3 * 3, 5 * 5, 7 * 7, 1 * 1, 3 * 3, 5 * 5, 7 * 7);

        assert_eq!(d, transmute(vec_mulo(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mulo_signed_char() {
        let a: vector_signed_char = transmute(i8x16::new(
            0, 1, -2, 3, -4, 5, -6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
        ));
        let d = i16x8::new(1 * 1, 3 * 3, 5 * 5, 7 * 7, 1 * 1, 3 * 3, 5 * 5, 7 * 7);

        assert_eq!(d, transmute(vec_mulo(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mulo_unsigned_short() {
        let a: vector_unsigned_short = transmute(u16x8::new(0, 1, 2, 3, 4, 5, 6, 7));
        let d = u32x4::new(1 * 1, 3 * 3, 5 * 5, 7 * 7);

        assert_eq!(d, transmute(vec_mulo(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_mulo_signed_short() {
        let a: vector_signed_short = transmute(i16x8::new(0, 1, -2, 3, -4, 5, -6, 7));
        let d = i32x4::new(1 * 1, 3 * 3, 5 * 5, 7 * 7);

        assert_eq!(d, transmute(vec_mulo(a, a)));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn vec_add_i32x4_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        let y = i32x4::new(4, 3, 2, 1);
        let x: vector_signed_int = transmute(x);
        let y: vector_signed_int = transmute(y);
        let z = vec_add(x, y);
        assert_eq!(i32x4::splat(5), transmute(z));
    }

    #[simd_test(enable = "altivec")]
    unsafe fn vec_ctf_u32() {
        let v: vector_unsigned_int = transmute(u32x4::new(u32::MIN, u32::MAX, u32::MAX, 42));
        let v2 = vec_ctf::<1, _>(v);
        let r2: vector_float = transmute(f32x4::new(0.0, 2147483600.0, 2147483600.0, 21.0));
        let v4 = vec_ctf::<2, _>(v);
        let r4: vector_float = transmute(f32x4::new(0.0, 1073741800.0, 1073741800.0, 10.5));
        let v8 = vec_ctf::<3, _>(v);
        let r8: vector_float = transmute(f32x4::new(0.0, 536870900.0, 536870900.0, 5.25));

        let check = |a, b| {
            let r = transmute(vec_cmple(
                vec_abs(vec_sub(a, b)),
                vec_splats(std::f32::EPSILON),
            ));
            let e = m32x4::new(true, true, true, true);
            assert_eq!(e, r);
        };

        check(v2, r2);
        check(v4, r4);
        check(v8, r8);
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_ctu() {
        let v = u32x4::new(u32::MIN, u32::MAX, u32::MAX, 42);
        let v2: u32x4 = transmute(vec_ctu::<1>(transmute(f32x4::new(
            0.0,
            2147483600.0,
            2147483600.0,
            21.0,
        ))));
        let v4: u32x4 = transmute(vec_ctu::<2>(transmute(f32x4::new(
            0.0,
            1073741800.0,
            1073741800.0,
            10.5,
        ))));
        let v8: u32x4 = transmute(vec_ctu::<3>(transmute(f32x4::new(
            0.0,
            536870900.0,
            536870900.0,
            5.25,
        ))));

        assert_eq!(v2, v);
        assert_eq!(v4, v);
        assert_eq!(v8, v);
    }

    #[simd_test(enable = "altivec")]
    unsafe fn vec_ctf_i32() {
        let v: vector_signed_int = transmute(i32x4::new(i32::MIN, i32::MAX, i32::MAX - 42, 42));
        let v2 = vec_ctf::<1, _>(v);
        let r2: vector_float =
            transmute(f32x4::new(-1073741800.0, 1073741800.0, 1073741800.0, 21.0));
        let v4 = vec_ctf::<2, _>(v);
        let r4: vector_float = transmute(f32x4::new(-536870900.0, 536870900.0, 536870900.0, 10.5));
        let v8 = vec_ctf::<3, _>(v);
        let r8: vector_float = transmute(f32x4::new(-268435460.0, 268435460.0, 268435460.0, 5.25));

        let check = |a, b| {
            let r = transmute(vec_cmple(
                vec_abs(vec_sub(a, b)),
                vec_splats(std::f32::EPSILON),
            ));
            println!("{:?} {:?}", a, b);
            let e = m32x4::new(true, true, true, true);
            assert_eq!(e, r);
        };

        check(v2, r2);
        check(v4, r4);
        check(v8, r8);
    }

    #[simd_test(enable = "altivec")]
    unsafe fn test_vec_cts() {
        let v = i32x4::new(i32::MIN, i32::MAX, i32::MAX, 42);
        let v2: i32x4 = transmute(vec_cts::<1>(transmute(f32x4::new(
            -1073741800.0,
            1073741800.0,
            1073741800.0,
            21.0,
        ))));
        let v4: i32x4 = transmute(vec_cts::<2>(transmute(f32x4::new(
            -536870900.0,
            536870900.0,
            536870900.0,
            10.5,
        ))));
        let v8: i32x4 = transmute(vec_cts::<3>(transmute(f32x4::new(
            -268435460.0,
            268435460.0,
            268435460.0,
            5.25,
        ))));

        assert_eq!(v2, v);
        assert_eq!(v4, v);
        assert_eq!(v8, v);
    }
}
