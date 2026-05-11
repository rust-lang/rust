//! LoongArch64 SIMD helpers

use self as ls;
use crate::intrinsics::simd as is;

// Internal extension trait for concrete `Simd<T, N>` types.
//
// Provides a small set of helper functionality (`Elem` and `splat`)
// so generic and macro-based code can operate on different SIMD
// vector types in a uniform way.
pub(super) const trait SimdExt: Sized {
    type Elem;

    unsafe fn splat(v: i64) -> Self;
}

macro_rules! impl_simd_ext {
    ($v:ident, $e:ty) => {
        #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
        impl const SimdExt for crate::core_arch::simd::$v {
            type Elem = $e;

            #[inline(always)]
            unsafe fn splat(v: i64) -> Self {
                is::simd_splat(v as Self::Elem)
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
impl_simd_ext!(u16x8, u16);
impl_simd_ext!(u16x16, u16);
impl_simd_ext!(i32x4, i32);
impl_simd_ext!(i32x8, i32);
impl_simd_ext!(u32x4, u32);
impl_simd_ext!(u32x8, u32);
impl_simd_ext!(i64x2, i64);
impl_simd_ext!(i64x4, i64);
impl_simd_ext!(u64x2, u64);
impl_simd_ext!(u64x4, u64);

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_andn<T: Copy + const SimdExt>(a: T, b: T) -> T {
    is::simd_and(ls::simd_not(a), b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_fmsub<T: Copy>(a: T, b: T, c: T) -> T {
    is::simd_fma(a, b, is::simd_neg(c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_fnmadd<T: Copy>(a: T, b: T, c: T) -> T {
    is::simd_neg(is::simd_fma(a, b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_fnmsub<T: Copy>(a: T, b: T, c: T) -> T {
    is::simd_neg(ls::simd_fmsub(a, b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_madd<T: Copy>(a: T, b: T, c: T) -> T {
    is::simd_add(a, is::simd_mul(b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_msub<T: Copy>(a: T, b: T, c: T) -> T {
    is::simd_sub(a, is::simd_mul(b, c))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_nor<T: Copy + const SimdExt>(a: T, b: T) -> T {
    ls::simd_not(is::simd_or(a, b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_not<T: Copy + const SimdExt>(a: T) -> T {
    is::simd_xor(a, ls::simd_splat(!0))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_orn<T: Copy + const SimdExt>(a: T, b: T) -> T {
    is::simd_or(a, ls::simd_not(b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_shl<T: Copy + const SimdExt>(a: T, b: T) -> T {
    let m = (size_of::<T::Elem>() * 8 - 1) as i64;
    is::simd_shl(a, is::simd_and(b, ls::simd_splat(m)))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_shr<T: Copy + const SimdExt>(a: T, b: T) -> T {
    let m = (size_of::<T::Elem>() * 8 - 1) as i64;
    is::simd_shr(a, is::simd_and(b, ls::simd_splat(m)))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_splat<T: Copy + const SimdExt>(a: i64) -> T {
    T::splat(a)
}

macro_rules! impl_vv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ty) => {
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
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $gty:ty) => {
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

macro_rules! impl_sv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $ibs:expr) => {
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
}

pub(super) use impl_sv;

macro_rules! impl_vvv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ty) => {
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
}

pub(super) use impl_vvv;

macro_rules! impl_vuv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, (size_of::<<$ity as SimdExt>::Elem>() * 8).ilog2());
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = ls::simd_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = ls::simd_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vuv;

macro_rules! impl_vug {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
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
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>(a: $oty) -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = ls::simd_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vsv;

macro_rules! impl_vvvv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ty) => {
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

macro_rules! impl_vugv {
    ($ft:literal, $name:ident, $op:path, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
        #[inline]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
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
