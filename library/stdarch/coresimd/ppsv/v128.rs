//! 128-bit wide portable packed vector types.

simd_api_imports!();

use ::coresimd::simd::{b8x2, b8x4, b8x8};

simd_i_ty! {
    i8x16: 16, i8, b8x16, i8x16_tests |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8  |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `i8` lanes.
}

simd_u_ty! {
    u8x16: 16, u8, b8x16, u8x16_tests |
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `u8` lanes.
}

simd_b_ty! {
    b8x16: 16, i8, b8x16_tests |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8  |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `bool` lanes.
}

simd_i_ty! {
    i16x8: 8, i16, b8x8, i16x8_tests |
    i16, i16, i16, i16, i16, i16, i16, i16 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 128-bit vector with 8 `i16` lanes.
}

simd_u_ty! {
    u16x8: 8, u16, b8x8, u16x8_tests |
    u16, u16, u16, u16, u16, u16, u16, u16 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 128-bit vector with 8 `u16` lanes.
}

simd_i_ty! {
    i32x4: 4, i32, b8x4, i32x4_tests |
    i32, i32, i32, i32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `i32` lanes.
}

simd_u_ty! {
    u32x4: 4, u32, b8x4, u32x4_tests |
    u32, u32, u32, u32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `u32` lanes.
}

simd_f_ty! {
    f32x4: 4, f32, b8x4, f32x4_tests |
    f32, f32, f32, f32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `f32` lanes.
}

simd_i_ty! {
    i64x2: 2, i64, b8x2, i64x2_tests |
    i64, i64 |
    x0, x1 |
    /// A 128-bit vector with 2 `u64` lanes.
}

simd_u_ty! {
    u64x2: 2, u64, b8x2, u64x2_tests |
    u64, u64 |
    x0, x1 |
    /// A 128-bit vector with 2 `u64` lanes.
}

simd_f_ty! {
    f64x2: 2, f64, b8x2, f64x2_tests |
    f64, f64 |
    x0, x1 |
    /// A 128-bit vector with 2 `f64` lanes.
}

impl_from_bits!(
    u64x2: u64,
    u64x2_from_bits | i64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    i64x2: i64,
    i64x2_from_bits | u64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    f64x2: f64,
    f64x2_from_bits | i64x2,
    u64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    u32x4: u32,
    u32x4_from_bits | u64x2,
    i64x2,
    f64x2,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    i32x4: i32,
    i32x4_from_bits | u64x2,
    i64x2,
    f64x2,
    u32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    f32x4: f32,
    f32x4_from_bits | u64x2,
    i64x2,
    f64x2,
    i32x4,
    u32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    u16x8: u16,
    u16x8_from_bits | u64x2,
    i64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    i16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    i16x8: i16,
    i16x8_from_bits | u64x2,
    i64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    u8x16,
    i8x16,
    b8x16
);
impl_from_bits!(
    u8x16: u8,
    u8x16_from_bits | u64x2,
    i64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    i8x16,
    b8x16
);
impl_from_bits!(
    i8x16: i8,
    i8x16_from_bits | u64x2,
    i64x2,
    f64x2,
    u32x4,
    i32x4,
    f32x4,
    u16x8,
    i16x8,
    u8x16,
    b8x16
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m128i;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m128d;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(f64x2: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u64x2: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i64x2: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(f32x4: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u32x4: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i32x4: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u16x8: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i16x8: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u8x16: __m128, __m128i, __m128d);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i8x16: __m128, __m128i, __m128d);

impl_from!(
    f64x2: f64,
    f64x2_from | f32x2,
    u64x2,
    i64x2,
    u32x2,
    i32x2,
    u16x2,
    i16x2,
    u8x2,
    i8x2
);
impl_from!(
    f32x4: f32,
    f32x4_from | f64x4,
    u64x4,
    i64x4,
    u32x4,
    i32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    u64x2: u64,
    u64x2_from | f32x2,
    f64x2,
    i64x2,
    i32x2,
    u32x2,
    i16x2,
    u16x2,
    i8x2,
    u8x2
);
impl_from!(
    i64x2: i64,
    i64x2_from | f32x2,
    f64x2,
    u64x2,
    i32x2,
    u32x2,
    i16x2,
    u16x2,
    i8x2,
    u8x2
);
impl_from!(
    u32x4: u32,
    u32x4_from | f64x4,
    u64x4,
    i64x4,
    f32x4,
    i32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    i32x4: i32,
    i32x4_from | f64x4,
    u64x4,
    i64x4,
    f32x4,
    u32x4,
    u16x4,
    i16x4,
    u8x4,
    i8x4
);
impl_from!(
    i16x8: i16,
    i16x8_from | f64x8,
    u64x8,
    i64x8,
    f32x8,
    u32x8,
    i32x8,
    u16x8,
    u8x8,
    i8x8
);
impl_from!(
    u16x8: u16,
    u16x8_from | f64x8,
    u64x8,
    i64x8,
    f32x8,
    u32x8,
    i32x8,
    i16x8,
    u8x8,
    i8x8
);
