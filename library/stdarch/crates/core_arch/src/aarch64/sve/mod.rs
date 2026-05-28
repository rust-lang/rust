//! SVE intrinsics

#![allow(non_camel_case_types)]

// `generated.rs` has a `super::*` and this import is for that
use crate::intrinsics::{simd::*, *};

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
#[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
pub use self::generated::*;

use crate::{marker::ConstParamTy, mem::transmute};

pub(super) trait AsUnsigned {
    type Unsigned;
    unsafe fn as_unsigned(self) -> Self::Unsigned;
}

pub(super) trait AsSigned {
    type Signed;
    unsafe fn as_signed(self) -> Self::Signed;
}

/// Same as `Into` but with into being unsafe so that it can have the required `target_feature`
pub(super) trait SveInto<T>: Sized {
    unsafe fn sve_into(self) -> T;
}

macro_rules! impl_sve_type {
    ($(($v:vis, $elem_type:ty, $name:ident, $elt:literal))*) => ($(
        #[doc = concat!("Scalable vector of type ", stringify!($elem_type))]
        #[derive(Clone, Copy, Debug)]
        #[rustc_scalable_vector($elt)]
        #[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
        $v struct $name($elem_type);
    )*)
}

macro_rules! impl_sve_tuple_type {
    ($(($v:vis, $vec_type:ty, $elt:tt, $name:ident))*) => ($(
        impl_sve_tuple_type!(@ ($v, $vec_type, $elt, $name));
    )*);
    (@ ($v:vis, $vec_type:ty, 2, $name:ident)) => (
        #[doc = concat!("Two-element tuple of scalable vectors of type ", stringify!($vec_type))]
        #[derive(Clone, Copy, Debug)]
        #[rustc_scalable_vector]
        #[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
        $v struct $name($vec_type, $vec_type);
    );
    (@ ($v:vis, $vec_type:ty, 3, $name:ident)) => (
        #[doc = concat!("Three-element tuple of scalable vectors of type ", stringify!($vec_type))]
        #[derive(Clone, Copy, Debug)]
        #[rustc_scalable_vector]
        #[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
        $v struct $name($vec_type, $vec_type, $vec_type);
    );
    (@ ($v:vis, $vec_type:ty, 4, $name:ident)) => (
        #[doc = concat!("Four-element tuple of scalable vectors of type ", stringify!($vec_type))]
        #[derive(Clone, Copy, Debug)]
        #[rustc_scalable_vector]
        #[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
        $v struct $name($vec_type, $vec_type, $vec_type, $vec_type);
    );
}

macro_rules! impl_sign_conversions_sv {
    ($(($signed:ty, $unsigned:ty))*) => ($(
        impl AsUnsigned for $signed {
            type Unsigned = $unsigned;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_unsigned(self) -> $unsigned {
                transmute_unchecked(self)
            }
        }

        impl AsSigned for $unsigned {
            type Signed = $signed;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_signed(self) -> $signed {
                transmute_unchecked(self)
            }
        }
    )*)
}

macro_rules! impl_sign_conversions {
    ($(($signed:ty, $unsigned:ty))*) => ($(
        impl AsUnsigned for $signed {
            type Unsigned = $unsigned;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_unsigned(self) -> $unsigned {
                transmute(self)
            }
        }

        impl AsSigned for $unsigned {
            type Signed = $signed;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_signed(self) -> $signed {
                transmute(self)
            }
        }
    )*)
}

/// LLVM requires the predicate lane count to be the same as the lane count
/// it's working with. However the ACLE only defines one bool type and the
/// instruction set doesn't have this distinction. As a result we have to
/// create these internal types so we can match the LLVM signature. Each of
/// these internal types can be converted to the public `svbool_t` type and
/// the `svbool_t` type can be converted into these.
macro_rules! impl_internal_sve_predicate {
    ($(($name:ident, $elt:literal))*) => ($(
        impl_sve_type! {
            (pub(super), bool, $name, $elt)
        }

        impl SveInto<svbool_t> for $name {
            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn sve_into(self) -> svbool_t {
                #[allow(improper_ctypes)]
                unsafe extern "C" {
                    #[cfg_attr(
                        target_arch = "aarch64",
                        link_name = concat!("llvm.aarch64.sve.convert.to.svbool.nxv", $elt, "i1")
                    )]
                    fn convert_to_svbool(b: $name) -> svbool_t;
                }
                unsafe { convert_to_svbool(self) }
            }
        }

        #[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
        impl SveInto<$name> for svbool_t {
            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn sve_into(self) -> $name {
                #[allow(improper_ctypes)]
                unsafe extern "C" {
                    #[cfg_attr(
                        target_arch = "aarch64",
                        link_name = concat!("llvm.aarch64.sve.convert.from.svbool.nxv", $elt, "i1")
                    )]
                    fn convert_from_svbool(b: svbool_t) -> $name;
                }
                unsafe { convert_from_svbool(self) }
            }
        }
    )*)
}

impl_sve_type! {
    (pub, bool, svbool_t, 16)

    (pub, i8, svint8_t, 16)
    (pub, u8, svuint8_t, 16)

    (pub, i16, svint16_t, 8)
    (pub, u16, svuint16_t, 8)
    (pub, f32, svfloat32_t, 4)
    (pub, i32, svint32_t, 4)
    (pub, u32, svuint32_t, 4)
    (pub, f64, svfloat64_t, 2)
    (pub, i64, svint64_t, 2)
    (pub, u64, svuint64_t, 2)

    // Internal types:
    (pub(super), i8, nxv2i8, 2)
    (pub(super), i8, nxv4i8, 4)
    (pub(super), i8, nxv8i8, 8)

    (pub(super), i16, nxv2i16, 2)
    (pub(super), i16, nxv4i16, 4)

    (pub(super), i32, nxv2i32, 2)

    (pub(super), u8, nxv2u8, 2)
    (pub(super), u8, nxv4u8, 4)
    (pub(super), u8, nxv8u8, 8)

    (pub(super), u16, nxv2u16, 2)
    (pub(super), u16, nxv4u16, 4)

    (pub(super), u32, nxv2u32, 2)
}

impl_sve_tuple_type! {
    (pub, svint8_t, 2, svint8x2_t)
    (pub, svuint8_t, 2, svuint8x2_t)
    (pub, svint16_t, 2, svint16x2_t)
    (pub, svuint16_t, 2, svuint16x2_t)
    (pub, svfloat32_t, 2, svfloat32x2_t)
    (pub, svint32_t, 2, svint32x2_t)
    (pub, svuint32_t, 2, svuint32x2_t)
    (pub, svfloat64_t, 2, svfloat64x2_t)
    (pub, svint64_t, 2, svint64x2_t)
    (pub, svuint64_t, 2, svuint64x2_t)

    (pub, svint8_t, 3, svint8x3_t)
    (pub, svuint8_t, 3, svuint8x3_t)
    (pub, svint16_t, 3, svint16x3_t)
    (pub, svuint16_t, 3, svuint16x3_t)
    (pub, svfloat32_t, 3, svfloat32x3_t)
    (pub, svint32_t, 3, svint32x3_t)
    (pub, svuint32_t, 3, svuint32x3_t)
    (pub, svfloat64_t, 3, svfloat64x3_t)
    (pub, svint64_t, 3, svint64x3_t)
    (pub, svuint64_t, 3, svuint64x3_t)

    (pub, svint8_t, 4, svint8x4_t)
    (pub, svuint8_t, 4, svuint8x4_t)
    (pub, svint16_t, 4, svint16x4_t)
    (pub, svuint16_t, 4, svuint16x4_t)
    (pub, svfloat32_t, 4, svfloat32x4_t)
    (pub, svint32_t, 4, svint32x4_t)
    (pub, svuint32_t, 4, svuint32x4_t)
    (pub, svfloat64_t, 4, svfloat64x4_t)
    (pub, svint64_t, 4, svint64x4_t)
    (pub, svuint64_t, 4, svuint64x4_t)
}

impl_sign_conversions! {
    (i8, u8)
    (i16, u16)
    (i32, u32)
    (i64, u64)
    (*const i8, *const u8)
    (*const i16, *const u16)
    (*const i32, *const u32)
    (*const i64, *const u64)
    (*mut i8, *mut u8)
    (*mut i16, *mut u16)
    (*mut i32, *mut u32)
    (*mut i64, *mut u64)
}

impl_sign_conversions_sv! {
    (svint8_t, svuint8_t)
    (svint16_t, svuint16_t)
    (svint32_t, svuint32_t)
    (svint64_t, svuint64_t)

    (svint8x2_t, svuint8x2_t)
    (svint16x2_t, svuint16x2_t)
    (svint32x2_t, svuint32x2_t)
    (svint64x2_t, svuint64x2_t)

    (svint8x3_t, svuint8x3_t)
    (svint16x3_t, svuint16x3_t)
    (svint32x3_t, svuint32x3_t)
    (svint64x3_t, svuint64x3_t)

    (svint8x4_t, svuint8x4_t)
    (svint16x4_t, svuint16x4_t)
    (svint32x4_t, svuint32x4_t)
    (svint64x4_t, svuint64x4_t)

    // Internal types:
    (nxv2i8, nxv2u8)
    (nxv4i8, nxv4u8)
    (nxv8i8, nxv8u8)

    (nxv2i16, nxv2u16)
    (nxv4i16, nxv4u16)

    (nxv2i32, nxv2u32)
}

impl_internal_sve_predicate! {
    (svbool2_t, 2)
    (svbool4_t, 4)
    (svbool8_t, 8)
}

/// Patterns returned by a `PTRUE`
#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
#[non_exhaustive]
#[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
pub enum svpattern {
    /// Activate the largest power-of-two number of elements that is less than the vector length
    SV_POW2 = 0,
    /// Activate the first element
    SV_VL1 = 1,
    /// Activate the first two elements
    SV_VL2 = 2,
    /// Activate the first three elements
    SV_VL3 = 3,
    /// Activate the first four elements
    SV_VL4 = 4,
    /// Activate the first five elements
    SV_VL5 = 5,
    /// Activate the first six elements
    SV_VL6 = 6,
    /// Activate the first seven elements
    SV_VL7 = 7,
    /// Activate the first eight elements
    SV_VL8 = 8,
    /// Activate the first sixteen elements
    SV_VL16 = 9,
    /// Activate the first thirty-two elements
    SV_VL32 = 10,
    /// Activate the first sixty-four elements
    SV_VL64 = 11,
    /// Activate the first one-hundred-and-twenty-eight elements
    SV_VL128 = 12,
    /// Activate the first two-hundred-and-fifty-six elements
    SV_VL256 = 13,
    /// Activate the largest multiple-of-four number of elements that is less than the vector length
    SV_MUL4 = 29,
    /// Activate the largest multiple-of-three number of elements that is less than the vector
    /// length
    SV_MUL3 = 30,
    /// Activate all elements
    SV_ALL = 31,
}

/// Addressing mode for prefetch intrinsics - allows the specification of the expected access
/// kind (read or write), the cache level to load the data, the data retention policy
/// (temporal or streaming)
#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
#[non_exhaustive]
#[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
pub enum svprfop {
    /// Temporal fetch of the addressed location for reading to the L1 cache (i.e. allocate in
    /// cache normally)
    SV_PLDL1KEEP = 0,
    /// Streaming fetch of the addressed location for reading to the L1 cache (i.e. memory only
    /// used once)
    SV_PLDL1STRM = 1,
    /// Temporal fetch of the addressed location for reading to the L2 cache (i.e. allocate in
    /// cache normally)
    SV_PLDL2KEEP = 2,
    /// Streaming fetch of the addressed location for reading to the L2 cache (i.e. memory only
    /// used once)
    SV_PLDL2STRM = 3,
    /// Temporal fetch of the addressed location for reading to the L3 cache (i.e. allocate in
    /// cache normally)
    SV_PLDL3KEEP = 4,
    /// Streaming fetch of the addressed location for reading to the L3 cache (i.e. memory only
    /// used once)
    SV_PLDL3STRM = 5,
    /// Temporal fetch of the addressed location for writing to the L1 cache (i.e. allocate in
    /// cache normally)
    SV_PSTL1KEEP = 8,
    /// Temporal fetch of the addressed location for writing to the L1 cache (i.e. memory only
    /// used once)
    SV_PSTL1STRM = 9,
    /// Temporal fetch of the addressed location for writing to the L2 cache (i.e. allocate in
    /// cache normally)
    SV_PSTL2KEEP = 10,
    /// Temporal fetch of the addressed location for writing to the L2 cache (i.e. memory only
    /// used once)
    SV_PSTL2STRM = 11,
    /// Temporal fetch of the addressed location for writing to the L3 cache (i.e. allocate in
    /// cache normally)
    SV_PSTL3KEEP = 12,
    /// Temporal fetch of the addressed location for writing to the L3 cache (i.e. memory only
    /// used once)
    SV_PSTL3STRM = 13,
}

#[cfg(test)]
#[path = "ld_st_tests_aarch64.rs"]
mod ld_st_tests;
