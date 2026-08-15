//! LoongArch64 SIMD helpers

use crate::intrinsics::simd::*;

// Internal extension trait for concrete `Simd<T, N>` types.
//
// Provides a small set of helper functionality (`Elem` and `splat`)
// so generic and macro-based code can operate on different SIMD
// vector types in a uniform way.
pub(super) const trait SimdExt: Sized {
    type Elem;

    unsafe fn splat(v: i64) -> Self;
}

#[rustfmt::skip] // FIXME: https://github.com/rust-lang/stdarch/pull/2133#issuecomment-4524350350
macro_rules! impl_simd_ext {
    ($v:ident, $e:ty) => {
        #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
        const impl SimdExt for crate::core_arch::simd::$v {
            type Elem = $e;

            #[inline(always)]
            unsafe fn splat(v: i64) -> Self {
                simd_splat(v as Self::Elem)
            }
        }
    };
}

impl_simd_ext!(i8x16, i8);
impl_simd_ext!(i8x32, i8);
impl_simd_ext!(u8x16, u8);
impl_simd_ext!(u8x32, u8);
impl_simd_ext!(i16x8, i16);
impl_simd_ext!(i16x16, i16);
impl_simd_ext!(i16x32, i16);
impl_simd_ext!(u16x8, u16);
impl_simd_ext!(u16x16, u16);
impl_simd_ext!(u16x32, u16);
impl_simd_ext!(i32x4, i32);
impl_simd_ext!(i32x8, i32);
impl_simd_ext!(i32x16, i32);
impl_simd_ext!(u32x4, u32);
impl_simd_ext!(u32x8, u32);
impl_simd_ext!(u32x16, u32);
impl_simd_ext!(i64x2, i64);
impl_simd_ext!(i64x4, i64);
impl_simd_ext!(i64x8, i64);
impl_simd_ext!(u64x2, u64);
impl_simd_ext!(u64x4, u64);
impl_simd_ext!(u64x8, u64);
impl_simd_ext!(i128x2, i128);
impl_simd_ext!(u128x2, u128);
impl_simd_ext!(i128x4, i128);
impl_simd_ext!(u128x4, u128);

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_abs<T: Copy + const SimdExt>(a: T) -> T {
    let m: T = simd_lt(a, simd_ext_splat(0));
    simd_select(m, simd_neg(a), a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_absd<T: Copy>(a: T, b: T) -> T {
    let m: T = simd_gt(a, b);
    simd_select(m, simd_sub(a, b), simd_sub(b, a))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_adda<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_add(simd_ext_abs(a), simd_ext_abs(b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_andn<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_and(simd_ext_not(a), b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bitclr<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_ext_andn(simd_ext_shl(simd_ext_splat(1), b), a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bitrev<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_xor(simd_ext_shl(simd_ext_splat(1), b), a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_bitset<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_or(simd_ext_shl(simd_ext_splat(1), b), a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_fmsub<T: Copy>(a: T, b: T, c: T) -> T {
    simd_fma(a, b, simd_neg(c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_fnmadd<T: Copy>(a: T, b: T, c: T) -> T {
    simd_neg(simd_fma(a, b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_fnmsub<T: Copy>(a: T, b: T, c: T) -> T {
    simd_neg(simd_ext_fmsub(a, b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_frecip_s<T: Copy>(a: T) -> T {
    simd_div(simd_splat(1.0f32), a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_frecip_d<T: Copy>(a: T) -> T {
    simd_div(simd_splat(1.0f64), a)
}

#[inline(always)]
pub(super) unsafe fn simd_ext_frsqrt_s<T: Copy>(a: T) -> T {
    simd_ext_frecip_s(simd_fsqrt(a))
}

#[inline(always)]
pub(super) unsafe fn simd_ext_frsqrt_d<T: Copy>(a: T) -> T {
    simd_ext_frecip_d(simd_fsqrt(a))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_ld<const I: i32, T: Copy>(a: *const i8) -> T {
    let a = a.offset(I as isize) as *const T;
    core::ptr::read_unaligned(a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_ldi<const I: i32, T: Copy + const SimdExt>() -> T {
    use crate::core_arch::simd::{i8x8, i16x4, i32x2};
    use crate::mem::transmute;

    #[inline(always)]
    const fn sext(v: i32, bits: u32) -> i64 {
        ((v as i64) << (64 - bits)) >> (64 - bits)
    }

    #[inline(always)]
    const fn expand_u8(imm: i32) -> u64 {
        let mut v = imm as u64;
        v = (v | (v << 28)) & 0x0000000f0000000f;
        v = (v | (v << 14)) & 0x0003000300030003;
        v = (v | (v << 7)) & 0x0101010101010101;
        v.wrapping_mul(0xff)
    }

    #[inline(always)]
    const fn fp32_imm(imm: i32) -> i32 {
        ((imm & 0x80) << 24)
            | ((!imm & 0x40) << 24)
            | (((imm & 0x40) >> 6).wrapping_mul(0x1f) << 25)
            | ((imm & 0x3f) << 19)
    }

    #[inline(always)]
    const fn fp64_imm(imm: i32) -> i64 {
        ((((imm & 0x80) << 24)
            | ((!imm & 0x40) << 24)
            | (((imm & 0x40) >> 6).wrapping_mul(0xff) << 22)
            | ((imm & 0x3f) << 16)) as i64)
            << 32
    }

    let imm8 = I & 0xff;
    let imm10 = I & 0x3ff;

    let r = if (I & 0x1000) == 0 {
        match (I >> 10) & 3 {
            0 => transmute(i8x8::splat(imm8 as i8)),
            1 => transmute(i16x4::splat(sext(imm10, 10) as i16)),
            2 => transmute(i32x2::splat(sext(imm10, 10) as i32)),
            3 => sext(imm10, 10),
            _ => unreachable!(),
        }
    } else {
        match (I >> 8) & 0xf {
            0..=3 => transmute(i32x2::splat(imm8 << (8 * ((I >> 8) & 3)))),
            4..=5 => transmute(i16x4::splat((imm8 as i16) << (8 * ((I >> 8) & 1)))),
            6 => transmute(i32x2::splat((imm8 << 8) | 0xff)),
            7 => transmute(i32x2::splat((imm8 << 16) | 0xffff)),
            8 => transmute(i8x8::splat(imm8 as i8)),
            9 => expand_u8(imm8) as i64,
            10 => transmute(i32x2::splat(fp32_imm(I))),
            11 => transmute(i32x2::from_array([fp32_imm(I), 0])),
            12 => fp64_imm(I),
            _ => unreachable!(),
        }
    };

    simd_ext_splat(r)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_ldx<T: Copy>(a: *const i8, b: i64) -> T {
    let a = a.offset(b as isize) as *const T;
    core::ptr::read_unaligned(a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_madd<T: Copy>(a: T, b: T, c: T) -> T {
    simd_add(a, simd_mul(b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_msub<T: Copy>(a: T, b: T, c: T) -> T {
    simd_sub(a, simd_mul(b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_muh<T: Copy, W: Copy + const SimdExt>(a: T, b: T) -> T {
    let a: W = simd_cast(a);
    let b: W = simd_cast(b);
    let p = simd_mul(a, b);
    simd_cast(simd_shr(
        p,
        simd_ext_splat((size_of::<W::Elem>() * 8 / 2) as i64),
    ))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_nor<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_ext_not(simd_or(a, b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_not<T: Copy + const SimdExt>(a: T) -> T {
    simd_xor(a, simd_ext_splat(!0))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_orn<T: Copy + const SimdExt>(a: T, b: T) -> T {
    simd_or(a, simd_ext_not(b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_rotr<T: Copy + const SimdExt>(a: T, b: T) -> T {
    let m = (size_of::<T::Elem>() * 8 - 1) as i64;
    let r = simd_and(b, simd_ext_splat(m));
    let l = simd_and(simd_sub(simd_ext_splat(m + 1), r), simd_ext_splat(m));
    simd_or(simd_shr(a, r), simd_shl(a, l))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_shl<T: Copy + const SimdExt>(a: T, b: T) -> T {
    let m = (size_of::<T::Elem>() * 8 - 1) as i64;
    simd_shl(a, simd_and(b, simd_ext_splat(m)))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_shr<T: Copy + const SimdExt>(a: T, b: T) -> T {
    let m = (size_of::<T::Elem>() * 8 - 1) as i64;
    simd_shr(a, simd_and(b, simd_ext_splat(m)))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_splat<T: Copy + const SimdExt>(a: i64) -> T {
    T::splat(a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_st<const I: i32, T: Copy>(a: T, b: *mut i8) {
    let b = b.offset(I as isize) as *mut T;
    core::ptr::write_unaligned(b, a);
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_ext_stx<T: Copy>(a: T, b: *mut i8, c: i64) {
    let b = b.offset(c as isize) as *mut T;
    core::ptr::write_unaligned(b, a);
}

macro_rules! impl_vv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op(a);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vv;

macro_rules! impl_gv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $gty) -> $oty {
            unsafe {
                let r: $ity = $op(a.into());
                transmute(r)
            }
        }
    };
}

pub(super) use impl_gv;

macro_rules! impl_ggv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $xty:ty, unsafe) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub unsafe fn $name(a: $gty, b: $xty) -> $oty {
            let r: $ity = $op(a, b);
            transmute(r)
        }
    };
}

pub(super) use impl_ggv;

macro_rules! impl_gsv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr, const, unsafe) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub unsafe fn $name<const IMM: i32>(a: $gty) -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            let r: $ity = $op::<IMM, _>(a);
            transmute(r)
        }
    };
}

pub(super) use impl_gsv;

macro_rules! impl_sv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(0)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>() -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let r: $ity = $op(IMM.into());
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr, const) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(0)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>() -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let r: $ity = $op::<IMM, _>();
                transmute(r)
            }
        }
    };
}

pub(super) use impl_sv;

macro_rules! impl_vvv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $wty:ty) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let r: $ity = $op::<$ity, $wty>(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvv;

macro_rules! impl_vgg {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $xty:ty, unsafe) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub unsafe fn $name(a: $oty, b: $gty, c: $xty) {
            $op(a, b, c);
        }
    };
}

pub(super) use impl_vgg;

macro_rules! impl_vgs {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr, const, unsafe) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(2)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub unsafe fn $name<const IMM: i32>(a: $oty, b: $gty) {
            static_assert_simm_bits!(IMM, $ibs);
            $op::<IMM, _>(a, b);
        }
    };
}

pub(super) use impl_vgs;

macro_rules! impl_vuv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, (size_of::<<$ity as SimdExt>::Elem>() * 8).ilog2());
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_ext_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_ext_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr, const) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op::<IMM, _>(a);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vuv;

macro_rules! impl_vug {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $gty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: <$ity as SimdExt>::Elem = $op(a, IMM);
                r as $gty
            }
        }
    };
}

pub(super) use impl_vug;

macro_rules! impl_vsv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>(a: $oty) -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_ext_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vsv;

macro_rules! impl_vvvv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty, c: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let c: $ity = transmute(c);
                let r: $ity = $op(a, b, c);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvvv;

macro_rules! impl_vvuv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr, const) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(2)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty, b: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let r: $ity = $op::<IMM, _>(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvuv;

macro_rules! impl_vugv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(2)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty, b: $gty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op(a, IMM, b as <$ity as SimdExt>::Elem);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vugv;
