#![allow(non_camel_case_types)]

// 导入父模块中的 simd_cast 函数
use super::simd_cast;

// ============================================================================
// 核心SVE类型定义 - 最小化版本用于编译测试
// ============================================================================

/// SVE谓词类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(1)]
#[repr(C)]
pub struct svbool_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool_t {
    fn clone(&self) -> Self { *self }
}

/// SVE双宽度谓词类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svbool2_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE四宽度谓词类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svbool4_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE八宽度谓词类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svbool8_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svbool8_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svbool8_t {
    fn clone(&self) -> Self { *self }
}

// ============================================================================
// SVE 向量类型定义
// ============================================================================

/// SVE 8位有符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint8_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位有符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint16_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位有符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svint32_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位有符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svint64_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位无符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint8_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位无符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint16_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位无符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svuint32_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位无符号整数向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svuint64_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位浮点向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svfloat32_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位浮点向量
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(2)]
#[repr(C)]
pub struct svfloat64_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位浮点向量 (使用 f32 作为底层类型)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat16_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16_t {
    fn clone(&self) -> Self { *self }
}

// ============================================================================
// SVE 向量元组类型定义
// ============================================================================

/// SVE 8位有符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svint8x2_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位无符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svuint8x2_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位有符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint16x2_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位无符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint16x2_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位浮点双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat32x2_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位有符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint32x2_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位无符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint32x2_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位浮点双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svfloat64x2_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位有符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svint64x2_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位无符号整数双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(4)]
#[repr(C)]
pub struct svuint64x2_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位有符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(48)]
#[repr(C)]
pub struct svint8x3_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位无符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(48)]
#[repr(C)]
pub struct svuint8x3_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位有符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svint16x3_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位无符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svuint16x3_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位浮点三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svfloat32x3_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位有符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svint32x3_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位无符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(12)]
#[repr(C)]
pub struct svuint32x3_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位浮点三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svfloat64x3_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位有符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svint64x3_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位无符号整数三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(6)]
#[repr(C)]
pub struct svuint64x3_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位有符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(64)]
#[repr(C)]
pub struct svint8x4_t(i8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint8x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint8x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 8位无符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(64)]
#[repr(C)]
pub struct svuint8x4_t(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint8x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint8x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位有符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svint16x4_t(i16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint16x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint16x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位无符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svuint16x4_t(u16);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint16x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint16x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位浮点四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svfloat32x4_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat32x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat32x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位有符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svint32x4_t(i32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint32x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint32x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 32位无符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svuint32x4_t(u32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint32x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint32x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位浮点四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svfloat64x4_t(f64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat64x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat64x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位有符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svint64x4_t(i64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svint64x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svint64x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 64位无符号整数四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(8)]
#[repr(C)]
pub struct svuint64x4_t(u64);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svuint64x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svuint64x4_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位浮点双向量 (x2)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(16)]
#[repr(C)]
pub struct svfloat16x2_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x2_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x2_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位浮点三向量 (x3)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(24)]
#[repr(C)]
pub struct svfloat16x3_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x3_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x3_t {
    fn clone(&self) -> Self { *self }
}

/// SVE 16位浮点四向量 (x4)
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[rustc_scalable_vector(32)]
#[repr(C)]
pub struct svfloat16x4_t(f32);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Copy for svfloat16x4_t {}
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl Clone for svfloat16x4_t {
    fn clone(&self) -> Self { *self }
}

// ============================================================================
// SVE 辅助类型
// ============================================================================

/// SVE模式类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, core::marker::ConstParamTy)]
pub struct svpattern(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svpattern {
    /// 从原始字节创建模式值
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn from_raw(value: u8) -> Self {
        svpattern(value)
    }

    /// 以原始字节形式返回模式值
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

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

/// SVE预取操作类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, core::marker::ConstParamTy)]
pub struct svprfop(u8);

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svprfop {
    /// 从原始字节创建预取操作值
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn from_raw(value: u8) -> Self {
        svprfop(value)
    }

    /// 以原始字节形式返回预取操作值
    #[inline(always)]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub const fn as_raw(self) -> u8 {
        self.0
    }
}

// ============================================================================
// 类型转换辅助函数（仅用于内部）
// ============================================================================
// 注意：simd_cast 函数定义在父模块 mod.rs 中，使用 transmute_copy 避免 E0511 错误

// ============================================================================
// 类 From trait 的转换方法 - 适用于交叉编译
// ============================================================================

// 方案：使用 Associated Functions 提供类似 From::from 的 API
// 这些方法添加了 #[target_feature(enable = "sve")]，可以在交叉编译时正常工作

/// svbool_t 的转换方法
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool_t {
    /// 转换为 svbool2_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// 转换为 svbool4_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// 转换为 svbool8_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }
}

/// svbool2_t 的转换方法
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool2_t {
    /// 从 svbool_t 创建（类似 From::from）
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// 转换为 svbool_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// 转换为 svbool4_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// 转换为 svbool8_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }

    /// 从 svbool4_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool4(x: svbool4_t) -> Self {
        simd_cast(x)
    }

    /// 从 svbool8_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool8(x: svbool8_t) -> Self {
        simd_cast(x)
    }
}

/// svbool4_t 的转换方法
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool4_t {
    /// 从 svbool_t 创建（类似 From::from）
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// 转换为 svbool_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// 转换为 svbool2_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// 转换为 svbool8_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool8(self) -> svbool8_t {
        simd_cast(self)
    }

    /// 从 svbool2_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool2(x: svbool2_t) -> Self {
        simd_cast(x)
    }

    /// 从 svbool8_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool8(x: svbool8_t) -> Self {
        simd_cast(x)
    }
}

/// svbool8_t 的转换方法
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl svbool8_t {
    /// 从 svbool_t 创建（类似 From::from）
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool(x: svbool_t) -> Self {
        simd_cast(x)
    }

    /// 转换为 svbool_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool(self) -> svbool_t {
        simd_cast(self)
    }

    /// 转换为 svbool2_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool2(self) -> svbool2_t {
        simd_cast(self)
    }

    /// 转换为 svbool4_t
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn into_svbool4(self) -> svbool4_t {
        simd_cast(self)
    }

    /// 从 svbool2_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool2(x: svbool2_t) -> Self {
        simd_cast(x)
    }

    /// 从 svbool4_t 创建
    #[inline]
    #[target_feature(enable = "sve")]
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    pub unsafe fn from_svbool4(x: svbool4_t) -> Self {
        simd_cast(x)
    }
}

// ============================================================================
// 类型转换 Trait - 用于生成的代码
// ============================================================================

/// 转换为无符号向量类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub trait AsUnsigned {
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    type Unsigned;
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    fn as_unsigned(self) -> Self::Unsigned;
}

/// 转换为有符号向量类型
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub trait AsSigned {
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    type Signed;
    #[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
    fn as_signed(self) -> Self::Signed;
}

// 为所有 SVE 整数类型实现转换 trait
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8_t {
    type Unsigned = svuint8_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16_t {
    type Unsigned = svuint16_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32_t {
    type Unsigned = svuint32_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64_t {
    type Unsigned = svuint64_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x2_t {
    type Unsigned = svuint8x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x2_t {
    type Unsigned = svuint16x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x2_t {
    type Unsigned = svuint32x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x2_t {
    type Unsigned = svuint64x2_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x3_t {
    type Unsigned = svuint8x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x3_t {
    type Unsigned = svuint16x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x3_t {
    type Unsigned = svuint32x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x3_t {
    type Unsigned = svuint64x3_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint8x4_t {
    type Unsigned = svuint8x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint16x4_t {
    type Unsigned = svuint16x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint32x4_t {
    type Unsigned = svuint32x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
impl AsUnsigned for svuint64x4_t {
    type Unsigned = svuint64x4_t;
    #[inline(always)]
    fn as_unsigned(self) -> Self::Unsigned { self }
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
    fn as_signed(self) -> Self::Signed { self }
}

// ============================================================================
