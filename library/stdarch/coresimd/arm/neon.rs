//! ARMv7 NEON intrinsics

use coresimd::simd::*;
use coresimd::simd_llvm::*;
#[cfg(test)]
use stdsimd_test::assert_instr;

types! {
    /// ARM-specific 64-bit wide vector of eight packed `i8`.
    pub struct int8x8_t(i8, i8, i8, i8, i8, i8, i8, i8);
    /// ARM-specific 64-bit wide vector of eight packed `u8`.
    pub struct uint8x8_t(u8, u8, u8, u8, u8, u8, u8, u8);
    /// ARM-specific 64-bit wide polynomial vector of eight packed `u8`.
    pub struct poly8x8_t(u8, u8, u8, u8, u8, u8, u8, u8);
    /// ARM-specific 64-bit wide vector of four packed `i16`.
    pub struct int16x4_t(i16, i16, i16, i16);
    /// ARM-specific 64-bit wide vector of four packed `u16`.
    pub struct uint16x4_t(u16, u16, u16, u16);
    // FIXME: ARM-specific 64-bit wide vector of four packed `f16`.
    // pub struct float16x4_t(f16, f16, f16, f16);
    /// ARM-specific 64-bit wide vector of four packed `u16`.
    pub struct poly16x4_t(u16, u16, u16, u16);
    /// ARM-specific 64-bit wide vector of two packed `i32`.
    pub struct int32x2_t(i32, i32);
    /// ARM-specific 64-bit wide vector of two packed `u32`.
    pub struct uint32x2_t(u32, u32);
    /// ARM-specific 64-bit wide vector of two packed `f32`.
    pub struct float32x2_t(f32, f32);
    /// ARM-specific 64-bit wide vector of one packed `i64`.
    pub struct int64x1_t(i64);
    /// ARM-specific 64-bit wide vector of one packed `u64`.
    pub struct uint64x1_t(u64);

    /// ARM-specific 128-bit wide vector of sixteem packed `i8`.
    pub struct int8x16_t(
        i8, i8 ,i8, i8, i8, i8 ,i8, i8,
        i8, i8 ,i8, i8, i8, i8 ,i8, i8,
    );
    /// ARM-specific 128-bit wide vector of sixteen packed `u8`.
    pub struct uint8x16_t(
        u8, u8 ,u8, u8, u8, u8 ,u8, u8,
        u8, u8 ,u8, u8, u8, u8 ,u8, u8,
    );
    /// ARM-specific 128-bit wide vector of sixteen packed `u8`.
    pub struct poly8x16_t(
        u8, u8, u8, u8, u8, u8, u8, u8,
        u8, u8, u8, u8, u8, u8, u8, u8
    );
    /// ARM-specific 128-bit wide vector of eight packed `i16`.
    pub struct int16x8_t(i16, i16, i16, i16, i16, i16, i16, i16);
    /// ARM-specific 128-bit wide vector of eight packed `u16`.
    pub struct uint16x8_t(u16, u16, u16, u16, u16, u16, u16, u16);
    // FIXME: ARM-specific 128-bit wide vector of eight packed `f16`.
    // pub struct float16x8_t(f16, f16, f16, f16, f16, f16, f16);
    /// ARM-specific 128-bit wide vector of eight packed `u16`.
    pub struct poly16x8_t(u16, u16, u16, u16, u16, u16, u16, u16);
    /// ARM-specific 128-bit wide vector of four packed `i32`.
    pub struct int32x4_t(i32, i32, i32, i32);
    /// ARM-specific 128-bit wide vector of four packed `u32`.
    pub struct uint32x4_t(u32, u32, u32, u32);
    /// ARM-specific 128-bit wide vector of four packed `f32`.
    pub struct float32x4_t(f32, f32, f32, f32);
    /// ARM-specific 128-bit wide vector of two packed `i64`.
    pub struct int64x2_t(i64, i64);
    /// ARM-specific 128-bit wide vector of two packed `u64`.
    pub struct uint64x2_t(u64, u64);
}

impl_from_bits_!(
    int8x8_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    uint8x8_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    int16x4_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    uint16x4_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    int32x2_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    uint32x2_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    int64x1_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    float32x2_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    poly8x8_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
impl_from_bits_!(
    poly16x4_t: u32x2,
    i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);

impl_from_bits_!(
    int8x16_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    uint8x16_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    poly8x16_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    int16x8_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    uint16x8_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    poly16x8_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    int32x4_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    uint32x4_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    float32x4_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    int64x2_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(
    uint64x2_t: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v2f32")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrsqrte.v2f32")]
    fn frsqrte_v2f32(a: float32x2_t) -> float32x2_t;

    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v8i8")]
    fn vpmins_v8i8(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v4i16")]
    fn vpmins_v4i16(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sminp.v2i32")]
    fn vpmins_v2i32(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v8i8")]
    fn vpminu_v8i8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v4i16")]
    fn vpminu_v4i16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpminu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uminp.v2i32")]
    fn vpminu_v2i32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmins.v2f32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminp.v2f32")]
    fn vpminf_v2f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;

    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v8i8")]
    fn vpmaxs_v8i8(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v4i16")]
    fn vpmaxs_v4i16(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smaxp.v2i32")]
    fn vpmaxs_v2i32(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v8i8")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v8i8")]
    fn vpmaxu_v8i8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v4i16")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v4i16")]
    fn vpmaxu_v4i16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxu.v2i32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umaxp.v2i32")]
    fn vpmaxu_v2i32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vpmaxs.v2f32")]
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxp.v2f32")]
    fn vpmaxf_v2f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(add))]
pub unsafe fn vaddq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fadd))]
pub unsafe fn vadd_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vadd))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fadd))]
pub unsafe fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    let a: int16x8_t = simd_cast(a);
    let b: int16x8_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    let a: int32x4_t = simd_cast(a);
    let b: int32x4_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(saddl))]
pub unsafe fn vaddl_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    let a: int64x2_t = simd_cast(a);
    let b: int64x2_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u8(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    let a: uint16x8_t = simd_cast(a);
    let b: uint16x8_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    let a: uint32x4_t = simd_cast(a);
    let b: uint32x4_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector long add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vaddl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uaddl))]
pub unsafe fn vaddl_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    let a: uint64x2_t = simd_cast(a);
    let b: uint64x2_t = simd_cast(b);
    simd_add(a, b)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s16(a: int16x8_t) -> int8x8_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s32(a: int32x4_t) -> int16x4_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_s64(a: int64x2_t) -> int32x2_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u16(a: uint16x8_t) -> uint8x8_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u32(a: uint32x4_t) -> uint16x4_t {
    simd_cast(a)
}

/// Vector narrow integer.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(xtn))]
pub unsafe fn vmovn_u64(a: uint64x2_t) -> uint32x2_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s8(a: int8x8_t) -> int16x8_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s16(a: int16x4_t) -> int32x4_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sxtl))]
pub unsafe fn vmovl_s32(a: int32x2_t) -> int64x2_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u8(a: uint8x8_t) -> uint16x8_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u16(a: uint16x4_t) -> uint32x4_t {
    simd_cast(a)
}

/// Vector long move.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmovl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uxtl))]
pub unsafe fn vmovl_u32(a: uint32x2_t) -> uint64x2_t {
    simd_cast(a)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frsqrte))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsqrte))]
pub unsafe fn vrsqrte_f32(a: float32x2_t) -> float32x2_t {
    frsqrte_v2f32(a)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s8 (a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vpmins_v8i8(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s16 (a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vpmins_v4i16(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sminp))]
pub unsafe fn vpmin_s32 (a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vpmins_v2i32(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u8 (a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vpminu_v8i8(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u16 (a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vpminu_v4i16(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uminp))]
pub unsafe fn vpmin_u32 (a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vpminu_v2i32(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fminp))]
pub unsafe fn vpmin_f32 (a: float32x2_t, b: float32x2_t) -> float32x2_t {
    vpminf_v2f32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s8 (a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vpmaxs_v8i8(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s16 (a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vpmaxs_v4i16(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smaxp))]
pub unsafe fn vpmax_s32 (a: int32x2_t, b: int32x2_t) -> int32x2_t {
    vpmaxs_v2i32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u8 (a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vpmaxu_v8i8(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u16 (a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    vpmaxu_v4i16(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umaxp))]
pub unsafe fn vpmax_u32 (a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    vpmaxu_v2i32(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vpmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmaxp))]
pub unsafe fn vpmax_f32 (a: float32x2_t, b: float32x2_t) -> float32x2_t {
    vpmaxf_v2f32(a, b)
}


#[cfg(test)]
mod tests {
    use coresimd::arm::*;
    use simd::*;
    use std::mem;
    use stdsimd_test::simd_test;

    #[simd_test = "neon"]
    unsafe fn test_vadd_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: i8x8 = vadd_s8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r: i8x16 = vaddq_s8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(8, 7, 6, 5);
        let e = i16x4::new(9, 9, 9, 9);
        let r: i16x4 = vadd_s16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: i16x8 = vaddq_s16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(8, 7);
        let e = i32x2::new(9, 9);
        let r: i32x2 = vadd_s32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(8, 7, 6, 5);
        let e = i32x4::new(9, 9, 9, 9);
        let r: i32x4 = vaddq_s32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: u8x8 = vadd_u8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r: u8x16 = vaddq_u8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(8, 7, 6, 5);
        let e = u16x4::new(9, 9, 9, 9);
        let r: u16x4 = vadd_u16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r: u16x8 = vaddq_u16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(8, 7);
        let e = u32x2::new(9, 9);
        let r: u32x2 = vadd_u32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(8, 7, 6, 5);
        let e = u32x4::new(9, 9, 9, 9);
        let r: u32x4 = vaddq_u32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vadd_f32() {
        let a = f32x2::new(1., 2.);
        let b = f32x2::new(8., 7.);
        let e = f32x2::new(9., 9.);
        let r: f32x2 = vadd_f32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddq_f32() {
        let a = f32x4::new(1., 2., 3., 4.);
        let b = f32x4::new(8., 7., 6., 5.);
        let e = f32x4::new(9., 9., 9., 9.);
        let r: f32x4 = vaddq_f32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_s8() {
        let v = ::std::i8::MAX;
        let a = i8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as i16);
        let e = i16x8::new(v, v, v, v, v, v, v, v);
        let r: i16x8 = vaddl_s8(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_s16() {
        let v = ::std::i16::MAX;
        let a = i16x4::new(v, v, v, v);
        let v = 2 * (v as i32);
        let e = i32x4::new(v, v, v, v);
        let r: i32x4 = vaddl_s16(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_s32() {
        let v = ::std::i32::MAX;
        let a = i32x2::new(v, v);
        let v = 2 * (v as i64);
        let e = i64x2::new(v, v);
        let r: i64x2 = vaddl_s32(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_u8() {
        let v = ::std::u8::MAX;
        let a = u8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as u16);
        let e = u16x8::new(v, v, v, v, v, v, v, v);
        let r: u16x8 = vaddl_u8(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_u16() {
        let v = ::std::u16::MAX;
        let a = u16x4::new(v, v, v, v);
        let v = 2 * (v as u32);
        let e = u32x4::new(v, v, v, v);
        let r: u32x4 = vaddl_u16(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vaddl_u32() {
        let v = ::std::u32::MAX;
        let a = u32x2::new(v, v);
        let v = 2 * (v as u64);
        let e = u64x2::new(v, v);
        let r: u64x2 = vaddl_u32(a.into_bits(), a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = vmovn_s16(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let e = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = vmovn_s32(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_s64() {
        let a = i64x2::new(1, 2);
        let e = i32x2::new(1, 2);
        let r: i32x2 = vmovn_s64(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = vmovn_u16(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let e = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = vmovn_u32(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovn_u64() {
        let a = u64x2::new(1, 2);
        let e = u32x2::new(1, 2);
        let r: u32x2 = vmovn_u64(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_s8() {
        let e = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = vmovl_s8(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_s16() {
        let e = i32x4::new(1, 2, 3, 4);
        let a = i16x4::new(1, 2, 3, 4);
        let r: i32x4 = vmovl_s16(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_s32() {
        let e = i64x2::new(1, 2);
        let a = i32x2::new(1, 2);
        let r: i64x2 = vmovl_s32(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_u8() {
        let e = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = vmovl_u8(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_u16() {
        let e = u32x4::new(1, 2, 3, 4);
        let a = u16x4::new(1, 2, 3, 4);
        let r: u32x4 = vmovl_u16(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vmovl_u32() {
        let e = u64x2::new(1, 2);
        let a = u32x2::new(1, 2);
        let r: u64x2 = vmovl_u32(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vrsqrt_f32() {
        let a = f32x2::new(1.0, 2.0);
        let e = f32x2::new(0.9980469, 0.7050781);
        let r: f32x2 = vrsqrte_f32(a.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_s8() {
        let a = i8x8::new(1, -2, 3, -4, 5, 6, 7, 8);
        let b = i8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i8x8::new(-2, -4, 5, 7, 0, 2, 4, 6);
        let r: i8x8 = vpmin_s8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_s16() {
        let a = i16x4::new(1, 2, 3, -4);
        let b = i16x4::new(0, 3, 2, 5);
        let e = i16x4::new(1, -4, 0, 2);
        let r: i16x4 = vpmin_s16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_s32() {
        let a = i32x2::new(1, -2);
        let b = i32x2::new(0, 3);
        let e = i32x2::new(-2, 0);
        let r: i32x2 = vpmin_s32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u8x8::new(1, 3, 5, 7, 0, 2, 4, 6);
        let r: u8x8 = vpmin_u8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 3, 2, 5);
        let e = u16x4::new(1, 3, 0, 2);
        let r: u16x4 = vpmin_u16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 3);
        let e = u32x2::new(1, 0);
        let r: u32x2 = vpmin_u32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmin_f32() {
        let a = f32x2::new(1., -2.);
        let b = f32x2::new(0., 3.);
        let e = f32x2::new(-2., 0.);
        let r: f32x2 = vpmin_f32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_s8() {
        let a = i8x8::new(1, -2, 3, -4, 5, 6, 7, 8);
        let b = i8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i8x8::new(1, 3, 6, 8, 3, 5, 7, 9);
        let r: i8x8 = vpmax_s8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_s16() {
        let a = i16x4::new(1, 2, 3, -4);
        let b = i16x4::new(0, 3, 2, 5);
        let e = i16x4::new(2, 3, 3, 5);
        let r: i16x4 = vpmax_s16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_s32() {
        let a = i32x2::new(1, -2);
        let b = i32x2::new(0, 3);
        let e = i32x2::new(1, 3);
        let r: i32x2 = vpmax_s32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u8x8::new(2, 4, 6, 8, 3, 5, 7, 9);
        let r: u8x8 = vpmax_u8(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(0, 3, 2, 5);
        let e = u16x4::new(2, 4, 3, 5);
        let r: u16x4 = vpmax_u16(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(0, 3);
        let e = u32x2::new(2, 3);
        let r: u32x2 = vpmax_u32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn test_vpmax_f32() {
        let a = f32x2::new(1., -2.);
        let b = f32x2::new(0., 3.);
        let e = f32x2::new(1., 3.);
        let r: f32x2 = vpmax_f32(a.into_bits(), b.into_bits()).into_bits();
        assert_eq!(r, e);
    }
}
