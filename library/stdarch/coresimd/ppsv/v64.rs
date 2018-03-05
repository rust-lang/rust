//! 64-bit wide portable packed vector types.

simd_api_imports!();

use ::coresimd::simd::{b8x4, b8x2};

simd_i_ty! {
    i8x8: 8, i8, b8x8, i8x8_tests |
    i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector with 8 `i8` lanes.
}

simd_u_ty! {
    u8x8: 8, u8, b8x8, u8x8_tests |
    u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector with 8 `u8` lanes.
}

simd_b_ty! {
    b8x8: 8, i8, b8x8_tests |
    i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector with 8 `bool` lanes.
}

simd_i_ty! {
    i16x4: 4, i16, b8x4, i16x4_tests |
    i16, i16, i16, i16 |
    x0, x1, x2, x3 |
    /// A 64-bit vector with 4 `i16` lanes.
}

simd_u_ty! {
    u16x4: 4, u16, b8x4, u16x4_tests |
    u16, u16, u16, u16 |
    x0, x1, x2, x3 |
    /// A 64-bit vector with 4 `u16` lanes.
}

simd_i_ty! {
    i32x2: 2, i32, b8x2, i32x2_tests |
    i32, i32 |
    x0, x1 |
    /// A 64-bit vector with 2 `i32` lanes.
}

simd_u_ty! {
    u32x2: 2, u32, b8x2, u32x2_tests |
    u32, u32 |
    x0, x1 |
    /// A 64-bit vector with 2 `u32` lanes.
}

simd_f_ty! {
    f32x2: 2, f32, b8x2, f32x2_tests |
    f32, f32 |
    x0, x1 |
    /// A 64-bit vector with 2 `f32` lanes.
}

impl_from_bits!(
    u32x2: u32,
    u32x2_from_bits | i32x2,
    f32x2,
    u16x4,
    i16x4,
    u8x8,
    i8x8,
    b8x8
);
impl_from_bits!(
    i32x2: i32,
    i32x2_from_bits | u32x2,
    f32x2,
    u16x4,
    i16x4,
    u8x8,
    i8x8,
    b8x8
);
impl_from_bits!(
    f32x2: f32,
    f32x2_from_bits | i32x2,
    u32x2,
    u16x4,
    i16x4,
    u8x8,
    i8x8,
    b8x8
);
impl_from_bits!(
    u16x4: u16,
    u16x4_from_bits | u32x2,
    i32x2,
    i16x4,
    u8x8,
    i8x8,
    b8x8
);
impl_from_bits!(
    i16x4: i16,
    i16x4_from_bits | u32x2,
    i32x2,
    u16x4,
    u8x8,
    i8x8,
    b8x8
);
impl_from_bits!(
    u8x8: u8,
    u8x8_from_bits | u32x2,
    i32x2,
    u16x4,
    i16x4,
    i8x8,
    b8x8
);
impl_from_bits!(
    i8x8: i8,
    i8x8_from_bits | u32x2,
    i32x2,
    u16x4,
    i16x4,
    u8x8,
    b8x8
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use coresimd::x86::__m64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(f32x2: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u32x2: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i32x2: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u16x4: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i16x4: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(u8x8: __m64);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl_from_bits_!(i8x8: __m64);

impl_from!(
    f32x2: f32,
    f32x2_from | f64x2,
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
    u32x2: u32,
    u32x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    i32x2,
    u16x2,
    i16x2,
    u8x2,
    i8x2
);

impl_from!(
    i32x2: i32,
    i32x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    u32x2,
    u16x2,
    i16x2,
    u8x2,
    i8x2
);

impl_from!(
    u16x4: u16,
    u16x4_from | f64x4,
    u64x4,
    i64x4,
    f32x4,
    i32x4,
    u32x4,
    i16x4,
    u8x4,
    i8x4
);

impl_from!(
    i16x4: i16,
    i16x4_from | f64x4,
    u64x4,
    i64x4,
    f32x4,
    i32x4,
    u32x4,
    u16x4,
    u8x4,
    i8x4
);
impl_from!(
    i8x8: i8,
    i8x8_from | f64x8,
    u64x8,
    i64x8,
    f32x8,
    u32x8,
    i32x8,
    i16x8,
    u16x8,
    u8x8
);
impl_from!(
    u8x8: u8,
    u8x8_from | f64x8,
    u64x8,
    i64x8,
    f32x8,
    u32x8,
    i32x8,
    i16x8,
    u16x8,
    i8x8
);
