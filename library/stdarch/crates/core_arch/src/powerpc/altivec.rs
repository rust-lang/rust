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

mod sealed {
    use super::*;

    macro_rules! test_impl {
        ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr:ident]) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $fun ($($v : $ty),*) -> $r {
                $call ($($v),*)
            }
        };
        ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr_altivec:ident / $instr_vsx:ident]) => {
            #[inline]
            #[target_feature(enable = "altivec")]
            #[cfg_attr(all(test, not(target_feature="vsx")), assert_instr($instr_altivec))]
            #[cfg_attr(all(test, target_feature="vsx"), assert_instr($instr_vsx))]
            pub unsafe fn $fun ($($v : $ty),*) -> $r {
                $call ($($v),*)
            }
        }

    }

    macro_rules! impl_vec_trait {
        ([$Trait:ident $m:ident] $fun:ident ($a:ty)) => {
            impl $Trait for $a {
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn $m(self) -> Self {
                    $fun(transmute(self))
                }
            }
        };
        ([$Trait:ident $m:ident] $fun:ident ($a:ty) -> $r:ty) => {
            impl $Trait for $a {
                type Result = $r;
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn $m(self) -> Self::Result {
                    $fun(transmute(self))
                }
            }
        };
        ([$Trait:ident $m:ident] 1 ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident, $sf: ident)) => {
            impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char) -> vector_signed_char }
            impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short) -> vector_signed_short }
            impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int) -> vector_signed_int }
            impl_vec_trait!{ [$Trait $m] $sf (vector_float) -> vector_float }
        };
        ([$Trait:ident $m:ident] $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
            impl $Trait<$b> for $a {
                type Result = $r;
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn $m(self, b: $b) -> Self::Result {
                    $fun(transmute(self), transmute(b))
                }
            }
        };
        ([$Trait:ident $m:ident] $fun:ident ($a:ty, ~$b:ty) -> $r:ty) => {
            impl_vec_trait!{ [$Trait $m] $fun ($a, $a) -> $r }
            impl_vec_trait!{ [$Trait $m] $fun ($a, $b) -> $r }
            impl_vec_trait!{ [$Trait $m] $fun ($b, $a) -> $r }
        };
        ([$Trait:ident $m:ident] ~($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
            impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, ~vector_bool_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, ~vector_bool_char) -> vector_signed_char }
            impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, ~vector_bool_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, ~vector_bool_short) -> vector_signed_short }
            impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, ~vector_bool_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, ~vector_bool_int) -> vector_signed_int }
        };
        ([$Trait:ident $m:ident] ~($fn:ident)) => {
            impl_vec_trait!{ [$Trait $m] ~($fn, $fn, $fn, $fn, $fn, $fn) }
        };
        ([$Trait:ident $m:ident] 2 ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
            impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, vector_signed_char) -> vector_signed_char }
            impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, vector_signed_short) -> vector_signed_short }
            impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, vector_signed_int) -> vector_signed_int }
        };
        ([$Trait:ident $m:ident] 2 ($fn:ident)) => {
            impl_vec_trait!{ [$Trait $m] ($fn, $fn, $fn, $fn, $fn, $fn) }
        }
    }

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
    unsafe fn andc(a: u8x16, b: u8x16) -> u8x16 {
        simd_and(simd_xor(u8x16::splat(0xff), b), a)
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
    unsafe fn mladd(a: i16x8, b: i16x8, c: i16x8) -> i16x8 {
        simd_add(simd_mul(a, b), c)
    }

    macro_rules! vector_mladd {
        ($a: ident, $bc: ident, $d: ident) => {
            impl VectorMladd<$bc> for $a {
                type Result = $d;
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mladd(self, b: $bc, c: $bc) -> Self::Result {
                    let a: i16x8 = transmute(self);
                    let b: i16x8 = transmute(b);
                    let c: i16x8 = transmute(c);

                    transmute(mladd(a, b, c))
                }
            }
        };
    }

    vector_mladd! { vector_unsigned_short, vector_unsigned_short, vector_unsigned_short }
    vector_mladd! { vector_unsigned_short, vector_signed_short, vector_signed_short }
    vector_mladd! { vector_signed_short, vector_unsigned_short, vector_signed_short }
    vector_mladd! { vector_signed_short, vector_signed_short, vector_signed_short }
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

/// Vector cmpb.
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

/// Vector add.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_add<T, U>(a: T, b: U) -> <T as sealed::VectorAdd<U>>::Result
where
    T: sealed::VectorAdd<U>,
{
    a.vec_add(b)
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
    [i32::min_value(), i32::max_value(), 1, -1],
    [-1, 1, 1, -1],
    [-1073741824, 1073741824, 1, -1] }

    test_vec_avg! { test_vec_avg_u32x4, u32x4,
    [u32::max_value(), 0, 1, 2],
    [2, 1, 0, 0],
    [2147483649, 1, 1, 1] }

    test_vec_avg! { test_vec_avg_i16x8, i16x8,
    [i16::min_value(), i16::max_value(), 1, -1, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0],
    [-16384, 16384, 1, -1, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_u16x8, u16x8,
    [u16::max_value(), 0, 1, 2, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0],
    [32769, 1, 1, 1, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_i8x16, i8x16,
    [i8::min_value(), i8::max_value(), 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-64, 64, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_avg! { test_vec_avg_u8x16, u8x16,
    [u8::max_value(), 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [129, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    macro_rules! test_vec_adds {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_adds, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_adds! { test_vec_adds_i32x4, i32x4,
    [i32::min_value(), i32::max_value(), 1, -1],
    [-1, 1, 1, -1],
    [i32::min_value(), i32::max_value(), 2, -2] }

    test_vec_adds! { test_vec_adds_u32x4, u32x4,
    [u32::max_value(), 0, 1, 2],
    [2, 1, 0, 0],
    [u32::max_value(), 1, 1, 2] }

    test_vec_adds! { test_vec_adds_i16x8, i16x8,
    [i16::min_value(), i16::max_value(), 1, -1, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0],
    [i16::min_value(), i16::max_value(), 2, -2, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_u16x8, u16x8,
    [u16::max_value(), 0, 1, 2, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0],
    [u16::max_value(), 1, 1, 2, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_i8x16, i8x16,
    [i8::min_value(), i8::max_value(), 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [i8::min_value(), i8::max_value(), 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_adds! { test_vec_adds_u8x16, u8x16,
    [u8::max_value(), 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [u8::max_value(), 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

    test_vec_2! { test_vec_addc, vec_addc, u32x4, [u32::max_value(), 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0] }

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
        let c: vector_signed_short =
            transmute(i16x8::new(0, 1, 2, 3, 4, 5, 6, i16::max_value() - 1));

        let d = i16x8::new(0, 3, 6, 9, 12, 15, 18, i16::max_value());

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
}
