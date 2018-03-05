//! 16-bit wide portable packed vector types.

simd_api_imports!();

simd_i_ty! {
    i8x2: 2, i8, b8x2, i8x2_tests |
    i8, i8 |
    x0, x1 |
    /// A 16-bit wide vector with 2 `i8` lanes.
}

simd_u_ty! {
    u8x2: 2, u8, b8x2, u8x2_tests |
    u8, u8 |
    x0, x1 |
    /// A 16-bit wide vector with 2 `u8` lanes.
}

simd_b_ty! {
    b8x2: 2, i8, b8x2_tests |
    i8, i8 |
    x0, x1 |
    /// A 16-bit wide vector with 2 `bool` lanes.
}

impl_from_bits!(i8x2: i8, i8x2_from_bits | u8x2, b8x2);
impl_from_bits!(u8x2: u8, u8x2_from_bits | i8x2, b8x2);

impl_from!(
    i8x2: i8,
    i8x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    u32x2,
    i32x2,
    u16x2,
    u8x2
);
impl_from!(
    u8x2: u8,
    u8x2_from | f64x2,
    u64x2,
    i64x2,
    f32x2,
    u32x2,
    i32x2,
    u16x2,
    i8x2
);
