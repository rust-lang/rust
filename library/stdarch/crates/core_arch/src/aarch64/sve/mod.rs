#![allow(unused_unsafe)]

// ============================================================================
// Module Declarations
// ============================================================================

mod sve;
mod sve2;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub mod types;

use types::*;

// ============================================================================
// Type Conversion Utilities
// ============================================================================

/// Bit-level reinterpretation for SVE types.
///
/// This function performs a pure bit-level reinterpretation. SVE wrapper types
/// are treated as opaque at this level to avoid triggering E0511 errors.
#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_reinterpret<T, U>(x: T) -> U {
    core::mem::transmute_copy::<T, U>(&x)
}

/// Type casting for SVE types.
///
/// Most SVE "casts" in stdarch are just layout-identical reinterpretations.
/// For actual value-semantic conversions, use the corresponding LLVM SVE convert
/// intrinsics in the specific API implementations.
#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_cast<T, U>(x: T) -> U {
    core::mem::transmute_copy::<T, U>(&x)
}

// ============================================================================
// SVE Select Operation (Predicated Selection)
// ============================================================================
//
// SVE's predicated selection uses LLVM's aarch64.sve.sel.* intrinsics.
// The intrinsic names correspond to element types/widths:
// - nxv16i8, nxv8i16, nxv4i32, nxv2i64 (integers)
// - nxv4f32, nxv2f64 (floats)
// - nxv16i1 (predicates)
//
// This approach avoids feeding non-SIMD types to simd_select, which would
// trigger E0511 errors.

/// Trait for static dispatch of SVE select operations to LLVM intrinsics.
pub(crate) trait __SveSelect {
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self;
}

// LLVM intrinsic declarations for SVE select operations
unsafe extern "C" {
    #[link_name = "llvm.aarch64.sve.sel.nxv16i8"]
    fn __llvm_sve_sel_nxv16i8(mask: svbool_t, a: svint8_t, b: svint8_t) -> svint8_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv8i16"]
    fn __llvm_sve_sel_nxv8i16(mask: svbool_t, a: svint16_t, b: svint16_t) -> svint16_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv4i32"]
    fn __llvm_sve_sel_nxv4i32(mask: svbool_t, a: svint32_t, b: svint32_t) -> svint32_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv2i64"]
    fn __llvm_sve_sel_nxv2i64(mask: svbool_t, a: svint64_t, b: svint64_t) -> svint64_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv4f32"]
    fn __llvm_sve_sel_nxv4f32(mask: svbool_t, a: svfloat32_t, b: svfloat32_t) -> svfloat32_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv2f64"]
    fn __llvm_sve_sel_nxv2f64(mask: svbool_t, a: svfloat64_t, b: svfloat64_t) -> svfloat64_t;

    #[link_name = "llvm.aarch64.sve.sel.nxv16i1"]
    fn __llvm_sve_sel_nxv16i1(mask: svbool_t, a: svbool_t, b: svbool_t) -> svbool_t;
}

// Implementation for signed integer types
impl __SveSelect for svint8_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv16i8(mask, a, b)
    }
}

impl __SveSelect for svint16_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv8i16(mask, a, b)
    }
}

impl __SveSelect for svint32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv4i32(mask, a, b)
    }
}

impl __SveSelect for svint64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv2i64(mask, a, b)
    }
}

// Implementation for unsigned integer types
// Note: svuint*_t and svint*_t share the same LLVM intrinsic at the same width
// since they have identical layouts in LLVM.
impl __SveSelect for svuint8_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv16i8(
            mask,
            core::mem::transmute(a),
            core::mem::transmute(b),
        ))
    }
}

impl __SveSelect for svuint16_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv8i16(
            mask,
            core::mem::transmute(a),
            core::mem::transmute(b),
        ))
    }
}

impl __SveSelect for svuint32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv4i32(
            mask,
            core::mem::transmute(a),
            core::mem::transmute(b),
        ))
    }
}

impl __SveSelect for svuint64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv2i64(
            mask,
            core::mem::transmute(a),
            core::mem::transmute(b),
        ))
    }
}

// Implementation for floating-point types
impl __SveSelect for svfloat32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv4f32(mask, a, b)
    }
}

impl __SveSelect for svfloat64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv2f64(mask, a, b)
    }
}

// Implementation for predicate type (1-bit predicate vector, nxv16i1)
impl __SveSelect for svbool_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv16i1(mask, a, b)
    }
}

// TODO: If f16/bf16/mfloat8 are supported in types.rs, add implementations:
// impl __SveSelect for svfloat16_t { ... }
// impl __SveSelect for svbfloat16_t { ... }
// impl __SveSelect for svmfloat8_t { ... }

// ============================================================================
// Predicate Type Conversions
// ============================================================================
//
// These implementations use transmute_copy for bit-level conversion.
// No target feature is required since transmute_copy is a pure bit-level
// operation that doesn't involve SVE instructions.

impl From<svbool2_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool2_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

impl From<svbool4_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool4_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

impl From<svbool8_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool8_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

// ============================================================================
// Public Select API
// ============================================================================
//
// This is the public entry point for select operations, maintaining the
// original function signature (called by sve/*.rs). It now uses trait-based
// static dispatch to LLVM SVE `sel` intrinsics instead of simd_select.

#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_select<M, T>(m: M, a: T, b: T) -> T
where
    M: Into<svbool_t>,
    T: __SveSelect,
{
    let mask: svbool_t = m.into();
    <T as __SveSelect>::sel(mask, a, b)
}

// ============================================================================
// Scalar Type Conversion Traits
// ============================================================================

/// Trait for converting between signed and unsigned scalar types.
trait ScalarConversion: Sized {
    type Unsigned;
    type Signed;
    fn as_unsigned(self) -> Self::Unsigned;
    fn as_signed(self) -> Self::Signed;
}

// Signed integer implementations
impl ScalarConversion for i8 {
    type Unsigned = u8;
    type Signed = i8;

    #[inline(always)]
    fn as_unsigned(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn as_signed(self) -> i8 {
        self
    }
}

impl ScalarConversion for i16 {
    type Unsigned = u16;
    type Signed = i16;

    #[inline(always)]
    fn as_unsigned(self) -> u16 {
        self as u16
    }

    #[inline(always)]
    fn as_signed(self) -> i16 {
        self
    }
}

impl ScalarConversion for i32 {
    type Unsigned = u32;
    type Signed = i32;

    #[inline(always)]
    fn as_unsigned(self) -> u32 {
        self as u32
    }

    #[inline(always)]
    fn as_signed(self) -> i32 {
        self
    }
}

impl ScalarConversion for i64 {
    type Unsigned = u64;
    type Signed = i64;

    #[inline(always)]
    fn as_unsigned(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn as_signed(self) -> i64 {
        self
    }
}

// Unsigned integer implementations
impl ScalarConversion for u8 {
    type Unsigned = u8;
    type Signed = i8;

    #[inline(always)]
    fn as_unsigned(self) -> u8 {
        self
    }

    #[inline(always)]
    fn as_signed(self) -> i8 {
        self as i8
    }
}

impl ScalarConversion for u16 {
    type Unsigned = u16;
    type Signed = i16;

    #[inline(always)]
    fn as_unsigned(self) -> u16 {
        self
    }

    #[inline(always)]
    fn as_signed(self) -> i16 {
        self as i16
    }
}

impl ScalarConversion for u32 {
    type Unsigned = u32;
    type Signed = i32;

    #[inline(always)]
    fn as_unsigned(self) -> u32 {
        self
    }

    #[inline(always)]
    fn as_signed(self) -> i32 {
        self as i32
    }
}

impl ScalarConversion for u64 {
    type Unsigned = u64;
    type Signed = i64;

    #[inline(always)]
    fn as_unsigned(self) -> u64 {
        self
    }

    #[inline(always)]
    fn as_signed(self) -> i64 {
        self as i64
    }
}

// ============================================================================
// Pointer Type Conversions
// ============================================================================

macro_rules! impl_scalar_conversion_for_ptr {
    ($(($unsigned:ty, $signed:ty)),*) => {
        $(
            impl ScalarConversion for *const $unsigned {
                type Unsigned = *const $unsigned;
                type Signed = *const $signed;

                #[inline(always)]
                fn as_unsigned(self) -> *const $unsigned {
                    self
                }

                #[inline(always)]
                fn as_signed(self) -> *const $signed {
                    self as *const $signed
                }
            }

            impl ScalarConversion for *const $signed {
                type Unsigned = *const $unsigned;
                type Signed = *const $signed;

                #[inline(always)]
                fn as_unsigned(self) -> *const $unsigned {
                    self as *const $unsigned
                }

                #[inline(always)]
                fn as_signed(self) -> *const $signed {
                    self
                }
            }

            impl ScalarConversion for *mut $unsigned {
                type Unsigned = *mut $unsigned;
                type Signed = *mut $signed;

                #[inline(always)]
                fn as_unsigned(self) -> *mut $unsigned {
                    self
                }

                #[inline(always)]
                fn as_signed(self) -> *mut $signed {
                    self as *mut $signed
                }
            }

            impl ScalarConversion for *mut $signed {
                type Unsigned = *mut $unsigned;
                type Signed = *mut $signed;

                #[inline(always)]
                fn as_unsigned(self) -> *mut $unsigned {
                    self as *mut $unsigned
                }

                #[inline(always)]
                fn as_signed(self) -> *mut $signed {
                    self
                }
            }
        )*
    };
}

impl_scalar_conversion_for_ptr!((u8, i8), (u16, i16), (u32, i32), (u64, i64));

// ============================================================================
// Public Exports
// ============================================================================

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve::*;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve2::*;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use types::*;

// ============================================================================
// LLVM Intrinsics and Public APIs
// ============================================================================

unsafe extern "C" {
    #[link_name = "llvm.aarch64.sve.whilelt"]
    fn __llvm_sve_whilelt_i32(i: i32, n: i32) -> svbool_t;
}

/// Generate a predicate for while less-than comparison.
///
/// Note: The svcntw() function is defined in sve.rs with the correct
/// LLVM intrinsic function signature.
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn svwhilelt_b32(i: i32, n: i32) -> svbool_t {
    __llvm_sve_whilelt_i32(i, n)
}
