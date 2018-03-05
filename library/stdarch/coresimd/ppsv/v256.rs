//! 256-bit wide portable packed vector types.

simd_api_imports!();

use ::coresimd::simd::{b8x16, b8x8, b8x4};

simd_i_ty! {
    i8x32: 32, i8, b8x32, i8x32_tests |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31 |
    /// A 256-bit vector with 32 `i8` lanes.
}

simd_u_ty! {
    u8x32: 32, u8, b8x32, u8x32_tests |
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31 |
    /// A 256-bit vector with 32 `u8` lanes.
}

simd_b_ty! {
    b8x32: 32, i8, b8x32_tests |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31 |
    /// A 256-bit vector with 32 `bool` lanes.
}

simd_i_ty! {
    i16x16: 16, i16, b8x16, i16x16_tests |
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 256-bit vector with 16 `i16` lanes.
}

simd_u_ty! {
    u16x16: 16, u16, b8x16, u16x16_tests |
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 256-bit vector with 16 `u16` lanes.
}

simd_i_ty! {
    i32x8: 8, i32, b8x8, i32x8_tests |
    i32, i32, i32, i32, i32, i32, i32, i32 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 256-bit vector with 8 `i32` lanes.
}

simd_u_ty! {
    u32x8: 8, u32, b8x8, u32x8_tests |
    u32, u32, u32, u32, u32, u32, u32, u32 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 256-bit vector with 8 `u32` lanes.
}

simd_f_ty! {
    f32x8: 8, f32, b8x8, f32x8_tests |
    f32, f32, f32, f32, f32, f32, f32, f32 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 256-bit vector with 8 `f32` lanes.
}

simd_i_ty! {
    i64x4: 4, i64, b8x4, i64x4_tests |
    i64, i64, i64, i64 |
    x0, x1, x2, x3 |
    /// A 256-bit vector with 4 `i64` lanes.
}

simd_u_ty! {
    u64x4: 4, u64, b8x4, u64x4_tests |
    u64, u64, u64, u64 |
    x0, x1, x2, x3 |
    /// A 256-bit vector with 4 `u64` lanes.
}

simd_f_ty! {
    f64x4: 4, f64, b8x4, f64x4_tests |
    f64, f64, f64, f64 |
    x0, x1, x2, x3 |
    /// A 256-bit vector with 4 `f64` lanes.
}

impl_from_bits!(
    i8x32: i8,
    i8x32_from_bits | u64x4,
    i64x4,
    f64x4,
    u32x8,
    i32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    b8x32
);
impl_from_bits!(
    u8x32: u8,
    u8x32_from_bits | u64x4,
    i64x4,
    f64x4,
    u32x8,
    i32x8,
    f32x8,
    u16x16,
    i16x16,
    i8x32,
    b8x32
);
impl_from_bits!(
    i16x16: i16,
    i16x16_from_bits | u64x4,
    i64x4,
    f64x4,
    u32x8,
    i32x8,
    f32x8,
    u16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    u16x16: u16,
    u16x16_from_bits | u64x4,
    i64x4,
    f64x4,
    u32x8,
    i32x8,
    f32x8,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    i32x8: i32,
    i32x8_from_bits | u64x4,
    i64x4,
    f64x4,
    u32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    u32x8: u32,
    u32x8_from_bits | u64x4,
    i64x4,
    f64x4,
    i32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    f32x8: f32,
    f32x8_from_bits | u64x4,
    i64x4,
    f64x4,
    i32x8,
    u32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    i64x4: i64,
    i64x4_from_bits | u64x4,
    f64x4,
    i32x8,
    u32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    u64x4: u64,
    u64x4_from_bits | i64x4,
    f64x4,
    i32x8,
    u32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);
impl_from_bits!(
    f64x4: f64,
    f64x4_from_bits | i64x4,
    u64x4,
    i32x8,
    u32x8,
    f32x8,
    u16x16,
    i16x16,
    u8x32,
    i8x32,
    b8x32
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m256;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m256i;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m256d;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(f64x4: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u64x4: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i64x4: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(f32x8: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u32x8: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i32x8: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u16x16: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i16x16: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u8x32: __m256, __m256i, __m256d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i8x32: __m256, __m256i, __m256d);

impl_from!(
    f64x4: f64,
    f64x4_from | u64x4,
    i64x4,
    u32x4,
    i32x4,
    f32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    i64x4: i64,
    i64x4_from | u64x4,
    f64x4,
    u32x4,
    i32x4,
    f32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    u64x4: u64,
    u64x4_from | i64x4,
    f64x4,
    u32x4,
    i32x4,
    f32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    f32x8: f32,
    f32x8_from | u64x8,
    i64x8,
    f64x8,
    u32x8,
    i32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);
impl_from!(
    i32x8: i32,
    i32x8_from | u64x8,
    i64x8,
    f64x8,
    u32x8,
    f32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);
impl_from!(
    u32x8: u32,
    u32x8_from | u64x8,
    i64x8,
    f64x8,
    i32x8,
    f32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);
impl_from!(
    i16x16: i16,
    i16x16_from | u32x16,
    i32x16,
    f32x16,
    u16x16,
    u8x16,
    i8x16
);
impl_from!(
    u16x16: u16,
    u16x16_from | u32x16,
    i32x16,
    f32x16,
    i16x16,
    u8x16,
    i8x16
);
impl_from!(i8x32: i8, i8x32_from | u16x32, i16x32, u8x32);
impl_from!(u8x32: u8, u8x32_from | u16x32, i16x32, i8x32);
