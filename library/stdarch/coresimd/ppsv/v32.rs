//! 32-bit wide portable packed vector types.

simd_api_imports!();
use coresimd::simd::b8x2;

simd_i_ty! {
    i16x2: 2, i16, b8x2, i16x2_tests |
    i16, i16 |
    x0, x1 |
    /// A 32-bit wide vector with 2 `i16` lanes.
}

simd_u_ty! {
    u16x2: 2, u16, b8x2, u16x2_tests |
    u16, u16 |
    x0, x1 |
    /// A 32-bit wide vector with 2 `u16` lanes.
}

simd_i_ty! {
    i8x4: 4, i8, b8x4, i8x4_tests |
    i8, i8, i8, i8  |
    x0, x1, x2, x3 |
    /// A 32-bit wide vector with 4 `i8` lanes.
}

simd_u_ty! {
    u8x4: 4, u8, b8x4, u8x4_tests |
    u8, u8, u8, u8  |
    x0, x1, x2, x3 |
    /// A 32-bit wide vector with 4 `u8` lanes.
}

simd_b_ty! {
    b8x4: 4, i8, b8x4_tests |
    i8, i8, i8, i8  |
    x0, x1, x2, x3 |
    /// A 32-bit wide vector with 4 `bool` lanes.
}

impl_from_bits!(i16x2: i16, i16x2_from_bits | u16x2, i8x4, u8x4, b8x4);
impl_from_bits!(u16x2: u16, u16x2_from_bits | i16x2, i8x4, u8x4, b8x4);
impl_from_bits!(i8x4: i8, i8x2_from_bits | i16x2, u16x2, u8x4, b8x4);
impl_from_bits!(u8x4: u8, u8x2_from_bits | i16x2, u16x2, i8x4, b8x4);

impl_from!(
    i16x2: i16,
    i16x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    u32x2,
    i32x2,
    u16x2,
    u8x2,
    i8x2
);

impl_from!(
    u16x2: u16,
    u16x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    u32x2,
    i32x2,
    i16x2,
    u8x2,
    i8x2
);

impl_from!(
    i8x4: i8,
    i8x4_from | f64x4,
    u64x4,
    i64x4,
    u32x4,
    i32x4,
    f32x4,
    u16x4,
    i16x4,
    u8x4
);

impl_from!(
    u8x4: u8,
    u8x4_from | f64x4,
    u64x4,
    i64x4,
    u32x4,
    i32x4,
    f32x4,
    u16x4,
    i16x4,
    i8x4
);
