#![allow(non_camel_case_types)]

// ============================================================================
// Imports
// ============================================================================

use super::simd_cast;

// ============================================================================
// SVE Predicate Types
// ============================================================================

/// SVE predicate type (1-bit predicate vector).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(1)]
#[repr(C)]
pub struct svbool_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE double-width predicate type (2-bit predicate vector).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svbool2_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE quad-width predicate type (4-bit predicate vector).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svbool4_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE octuple-width predicate type (8-bit predicate vector).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svbool8_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool8_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool8_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// Predicate Generation Functions
// ============================================================================
//
// These functions generate predicate vectors for loop control and conditional
// operations. They provide convenient wrappers around LLVM SVE intrinsics.

unsafe extern "C" {
    #[link_name = "llvm.aarch64.sve.whilelt"]
    fn __llvm_sve_whilelt_i32(i: i32, n: i32) -> svbool_t;
}

/// Generate a predicate for while less-than comparison.
///
/// This function generates a predicate vector where each element is true
/// if the corresponding index (starting from `i`) is less than `n`.
///
/// This is a convenience wrapper for loop control in SVE code. For more
/// specific variants (e.g., `svwhilelt_b32_s32`), see the functions in
/// the `sve` module.
///
/// # Safety
///
/// This function is marked unsafe because it requires the `sve` target feature.
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svwhilelt_b32(i: i32, n: i32) -> svbool_t {
    __llvm_sve_whilelt_i32(i, n)
}

// ============================================================================
// SVE Vector Types - Signed Integers
// ============================================================================

/// SVE 8-bit signed integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint8_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit signed integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint16_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit signed integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svint32_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit signed integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svint64_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Vector Types - Unsigned Integers
// ============================================================================

/// SVE 8-bit unsigned integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint8_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit unsigned integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint16_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit unsigned integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svuint32_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit unsigned integer vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svuint64_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Vector Types - Floating Point
// ============================================================================

/// SVE 32-bit floating-point vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svfloat32_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit floating-point vector.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svfloat64_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit floating-point vector (uses f32 as underlying type).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat16_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Vector Tuple Types - x2 (Double Vectors)
// ============================================================================

/// SVE 8-bit signed integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svint8x2_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 8-bit unsigned integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svuint8x2_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit signed integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint16x2_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit unsigned integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint16x2_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit signed integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint32x2_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit unsigned integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint32x2_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit signed integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svint64x2_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit unsigned integer double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svuint64x2_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit floating-point double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat32x2_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit floating-point double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svfloat64x2_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit floating-point double vector (x2).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svfloat16x2_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x2_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x2_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Vector Tuple Types - x3 (Triple Vectors)
// ============================================================================

/// SVE 8-bit signed integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(48)]
#[repr(C)]
pub struct svint8x3_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 8-bit unsigned integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(48)]
#[repr(C)]
pub struct svuint8x3_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit signed integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svint16x3_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit unsigned integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svuint16x3_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit signed integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svint32x3_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit unsigned integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svuint32x3_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit signed integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svint64x3_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit unsigned integer triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svuint64x3_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit floating-point triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svfloat32x3_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit floating-point triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svfloat64x3_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit floating-point triple vector (x3).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svfloat16x3_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x3_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x3_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Vector Tuple Types - x4 (Quadruple Vectors)
// ============================================================================

/// SVE 8-bit signed integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(64)]
#[repr(C)]
pub struct svint8x4_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 8-bit unsigned integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(64)]
#[repr(C)]
pub struct svuint8x4_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit signed integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svint16x4_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit unsigned integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svuint16x4_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit signed integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint32x4_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit unsigned integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint32x4_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit signed integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint64x4_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit unsigned integer quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint64x4_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 32-bit floating-point quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svfloat32x4_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 64-bit floating-point quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat64x4_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

/// SVE 16-bit floating-point quadruple vector (x4).
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svfloat16x4_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x4_t {}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x4_t {
    fn clone(&self) -> Self {
        *self
    }
}

// ============================================================================
// SVE Auxiliary Types
// ============================================================================

/// SVE pattern type for vector length specification.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, core::marker::ConstParamTy)]
pub struct svpattern(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svpattern {
    /// Create a pattern value from a raw byte.
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn from_raw(value: u8) -> Self {
        svpattern(value)
    }

    /// Return the pattern value as a raw byte.
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    // Pattern constants
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_ALL: svpattern = svpattern(31);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL1: svpattern = svpattern(1);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL2: svpattern = svpattern(2);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL3: svpattern = svpattern(3);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL4: svpattern = svpattern(4);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL5: svpattern = svpattern(5);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL6: svpattern = svpattern(6);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL7: svpattern = svpattern(7);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL8: svpattern = svpattern(8);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL16: svpattern = svpattern(9);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL32: svpattern = svpattern(10);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL64: svpattern = svpattern(11);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL128: svpattern = svpattern(12);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_VL256: svpattern = svpattern(13);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_POW2: svpattern = svpattern(30);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_MUL4: svpattern = svpattern(29);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_MUL3: svpattern = svpattern(28);
}

/// SVE prefetch operation type.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, core::marker::ConstParamTy)]
pub struct svprfop(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svprfop {
    /// Create a prefetch operation value from a raw byte.
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn from_raw(value: u8) -> Self {
        svprfop(value)
    }

    /// Return the prefetch operation value as a raw byte.
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    // Prefetch operation constants
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL1KEEP: svprfop = svprfop(0);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL1STRM: svprfop = svprfop(1);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL2KEEP: svprfop = svprfop(2);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL2STRM: svprfop = svprfop(3);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL3KEEP: svprfop = svprfop(4);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PLDL3STRM: svprfop = svprfop(5);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL1KEEP: svprfop = svprfop(8);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL1STRM: svprfop = svprfop(9);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL2KEEP: svprfop = svprfop(10);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL2STRM: svprfop = svprfop(11);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL3KEEP: svprfop = svprfop(12);
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const SV_PSTL3STRM: svprfop = svprfop(13);
}

// ============================================================================
// Predicate Type Conversion Methods
// ============================================================================
//
// These methods provide conversion APIs similar to From::from but with
// #[target_feature(enable = "sve")] for cross-compilation support.
// The simd_cast function is defined in the parent module (mod.rs) and uses
// transmute_copy to avoid E0511 errors.
//
// Note: These methods are organized by the source type for clarity.

/// Conversion methods for svbool_t.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool_t {
    /// Convert to svbool2_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// Convert to svbool4_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// Convert to svbool8_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }
}

/// Conversion methods for svbool2_t.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool2_t {
    /// Create from svbool_t (similar to From::from).
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// Convert to svbool_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// Convert to svbool4_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// Convert to svbool8_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }

    /// Create from svbool4_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool4(x: svbool4_t) -> Self {
        simd_cast(x)
    }

    /// Create from svbool8_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool8(x: svbool8_t) -> Self {
        simd_cast(x)
    }
}

/// Conversion methods for svbool4_t.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool4_t {
    /// Create from svbool_t (similar to From::from).
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// Convert to svbool_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// Convert to svbool2_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// Convert to svbool8_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }

    /// Create from svbool2_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool2(x: svbool2_t) -> Self {
        simd_cast(x)
    }

    /// Create from svbool8_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool8(x: svbool8_t) -> Self {
        simd_cast(x)
    }
}

/// Conversion methods for svbool8_t.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool8_t {
    /// Create from svbool_t (similar to From::from).
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// Convert to svbool_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// Convert to svbool2_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// Convert to svbool4_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// Create from svbool2_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool2(x: svbool2_t) -> Self {
        simd_cast(x)
    }

    /// Create from svbool4_t.
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool4(x: svbool4_t) -> Self {
        simd_cast(x)
    }
}

// ============================================================================
// From Trait Implementations for Predicate Types
// ============================================================================
//
// These implementations are used for .into() calls in generated code.
// Note: These implementations do not use target_feature because the From
// trait cannot have that attribute. The type conversion itself is safe and
// does not involve actual SIMD operations.

// Conversions from svbool_t to wider predicate types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool_t> for svbool2_t {
    #[inline(always)]
    fn from(x: svbool_t) -> Self {
        unsafe { simd_cast(x) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool_t> for svbool4_t {
    #[inline(always)]
    fn from(x: svbool_t) -> Self {
        unsafe { simd_cast(x) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool_t> for svbool8_t {
    #[inline(always)]
    fn from(x: svbool_t) -> Self {
        unsafe { simd_cast(x) }
    }
}

// Conversions from wider predicate types to svbool_t
// These implementations use transmute_copy for bit-level conversion.
// No target feature is required since transmute_copy is a pure bit-level
// operation that doesn't involve SVE instructions.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool2_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool2_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool4_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool4_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl From<svbool8_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool8_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

// ============================================================================
// Vector Type Conversion Traits
// ============================================================================
//
// These traits are used in generated code for converting between signed and
// unsigned vector types. They provide a consistent API for type conversions
// across all SVE vector types (single vectors and tuple vectors).

/// Trait for converting to unsigned vector types.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub trait AsUnsigned {
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    type Unsigned;
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    fn as_unsigned(self) -> Self::Unsigned;
}

/// Trait for converting to signed vector types.
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub trait AsSigned {
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    type Signed;
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    fn as_signed(self) -> Self::Signed;
}

// ============================================================================
// AsUnsigned and AsSigned Implementations - Single Vectors
// ============================================================================

// 8-bit types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8_t {
    type Unsigned = svuint8_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint8_t {
    type Signed = svint8_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint8_t {
    type Unsigned = svuint8_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint8_t {
    type Signed = svint8_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 16-bit types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16_t {
    type Unsigned = svuint16_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint16_t {
    type Signed = svint16_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint16_t {
    type Unsigned = svuint16_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint16_t {
    type Signed = svint16_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 32-bit types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32_t {
    type Unsigned = svuint32_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint32_t {
    type Signed = svint32_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint32_t {
    type Unsigned = svuint32_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint32_t {
    type Signed = svint32_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 64-bit types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64_t {
    type Unsigned = svuint64_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint64_t {
    type Signed = svint64_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint64_t {
    type Unsigned = svuint64_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint64_t {
    type Signed = svint64_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// ============================================================================
// AsUnsigned and AsSigned Implementations - x2 Tuple Vectors
// ============================================================================

// 8-bit x2 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x2_t {
    type Unsigned = svuint8x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint8x2_t {
    type Signed = svint8x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint8x2_t {
    type Unsigned = svuint8x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint8x2_t {
    type Signed = svint8x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 16-bit x2 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x2_t {
    type Unsigned = svuint16x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint16x2_t {
    type Signed = svint16x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint16x2_t {
    type Unsigned = svuint16x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint16x2_t {
    type Signed = svint16x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 32-bit x2 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x2_t {
    type Unsigned = svuint32x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint32x2_t {
    type Signed = svint32x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint32x2_t {
    type Unsigned = svuint32x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint32x2_t {
    type Signed = svint32x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 64-bit x2 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x2_t {
    type Unsigned = svuint64x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint64x2_t {
    type Signed = svint64x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint64x2_t {
    type Unsigned = svuint64x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint64x2_t {
    type Signed = svint64x2_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// ============================================================================
// AsUnsigned and AsSigned Implementations - x3 Tuple Vectors
// ============================================================================

// 8-bit x3 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x3_t {
    type Unsigned = svuint8x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint8x3_t {
    type Signed = svint8x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint8x3_t {
    type Unsigned = svuint8x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint8x3_t {
    type Signed = svint8x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 16-bit x3 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x3_t {
    type Unsigned = svuint16x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint16x3_t {
    type Signed = svint16x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint16x3_t {
    type Unsigned = svuint16x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint16x3_t {
    type Signed = svint16x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 32-bit x3 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x3_t {
    type Unsigned = svuint32x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint32x3_t {
    type Signed = svint32x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint32x3_t {
    type Unsigned = svuint32x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint32x3_t {
    type Signed = svint32x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 64-bit x3 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x3_t {
    type Unsigned = svuint64x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint64x3_t {
    type Signed = svint64x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint64x3_t {
    type Unsigned = svuint64x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint64x3_t {
    type Signed = svint64x3_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// ============================================================================
// AsUnsigned and AsSigned Implementations - x4 Tuple Vectors
// ============================================================================

// 8-bit x4 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x4_t {
    type Unsigned = svuint8x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint8x4_t {
    type Signed = svint8x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint8x4_t {
    type Unsigned = svuint8x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint8x4_t {
    type Signed = svint8x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 16-bit x4 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x4_t {
    type Unsigned = svuint16x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint16x4_t {
    type Signed = svint16x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint16x4_t {
    type Unsigned = svuint16x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint16x4_t {
    type Signed = svint16x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 32-bit x4 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x4_t {
    type Unsigned = svuint32x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint32x4_t {
    type Signed = svint32x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint32x4_t {
    type Unsigned = svuint32x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint32x4_t {
    type Signed = svint32x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// 64-bit x4 types
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x4_t {
    type Unsigned = svuint64x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        self
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svuint64x4_t {
    type Signed = svint64x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svint64x4_t {
    type Unsigned = svuint64x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned {
        unsafe { simd_cast(self) }
    }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsSigned for svint64x4_t {
    type Signed = svint64x4_t;
    #[inline(always)]
    fn as_signed(self) -> Self::Signed {
        self
    }
}

// ============================================================================
// LLVM Type Aliases
// ============================================================================
//
// These type aliases map LLVM machine representations (nxv* types) to Rust
// SVE types. They are used by the code generator to match LLVM intrinsic
// signatures with Rust type definitions.

// Signed integer type aliases
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv8i8 = svint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv4i8 = svint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv4i16 = svint16_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2i8 = svint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2i16 = svint16_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2i32 = svint32_t;

// Unsigned integer type aliases
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv8u8 = svuint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv4u8 = svuint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv4u16 = svuint16_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2u8 = svuint8_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2u16 = svuint16_t;

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub type nxv2u32 = svuint32_t;
