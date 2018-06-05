//! 16-bit wide portable packed vector types.

simd_i_ty! {
    i8x2: 2, i8, m8x2, i8x2_tests, test_v16 |
    i8, i8 |
    x0, x1 |
    /// A 16-bit wide vector with 2 `i8` lanes.
}

simd_u_ty! {
    u8x2: 2, u8, m8x2, u8x2_tests, test_v16 |
    u8, u8 |
    x0, x1 |
    /// A 16-bit wide vector with 2 `u8` lanes.
}

simd_m_ty! {
    m8x2: 2, i8, m8x2_tests, test_v16 |
    i8, i8 |
    x0, x1 |
    /// A 16-bit wide vector mask with 2 lanes.
}

impl_from_bits!(i8x2: i8, i8x2_from_bits, test_v16 | u8x2, m8x2);
impl_from_bits!(u8x2: u8, u8x2_from_bits, test_v16 | i8x2, m8x2);

impl_from!(
    i8x2: i8,
    i8x2_from,
    test_v16 | f64x2,
    u64x2,
    m64x2,
    i64x2,
    f32x2,
    u32x2,
    i32x2,
    m32x2,
    u16x2,
    m16x2,
    u8x2,
    m8x2
);
impl_from!(
    u8x2: u8,
    u8x2_from,
    test_v16 | f64x2,
    u64x2,
    i64x2,
    m64x2,
    f32x2,
    u32x2,
    i32x2,
    m32x2,
    u16x2,
    m16x2,
    i8x2,
    m8x2
);

impl_from!(m8x2: i8, m8x2_from, test_v16 | m64x2, m32x2, m16x2);
