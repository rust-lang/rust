//! 512-bit wide portable packed vector types.
use coresimd::simd::{b8x16, b8x32, b8x8};

simd_i_ty! {
    i8x64: 64, i8, b8x64, i8x64_tests, test_v512 |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63 |
    /// A 512-bit vector with 64 `i8` lanes.
}

simd_u_ty! {
    u8x64: 64, u8, b8x64, u8x64_tests, test_v512 |
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63 |
    /// A 512-bit vector with 64 `u8` lanes.
}

simd_b_ty! {
    b8x64: 64, i8, b8x64_tests, test_v512 |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63 |
    /// A 512-bit vector with 64 `bool` lanes.
}

simd_i_ty! {
    i16x32: 32, i16, b8x32, i16x32_tests, test_v512 |
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31 |
    /// A 512-bit vector with 32 `i16` lanes.
}

simd_u_ty! {
    u16x32: 32, u16, b8x32, u16x32_tests, test_v512 |
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31 |
    /// A 512-bit vector with 32 `u16` lanes.
}
simd_i_ty! {
    i32x16: 16, i32, b8x16, i32x16_tests, test_v512 |
    i32, i32, i32, i32, i32, i32, i32, i32,
    i32, i32, i32, i32, i32, i32, i32, i32 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 512-bit vector with 16 `i32` lanes.
}

simd_u_ty! {
    u32x16: 16, u32, b8x16, u32x16_tests, test_v512 |
    u32, u32, u32, u32, u32, u32, u32, u32,
    u32, u32, u32, u32, u32, u32, u32, u32 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 512-bit vector with 16 `u32` lanes.
}

simd_f_ty! {
    f32x16: 16, f32, b8x16, f32x16_tests, test_v512 |
    f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32 |
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 512-bit vector with 16 `f32` lanes.
}

simd_i_ty! {
    i64x8: 8, i64, b8x8, i64x8_tests, test_v512 |
    i64, i64, i64, i64, i64, i64, i64, i64 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 512-bit vector with 8 `i64` lanes.
}

simd_u_ty! {
    u64x8: 8, u64, b8x8, u64x8_tests, test_v512 |
    u64, u64, u64, u64, u64, u64, u64, u64 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 512-bit vector with 8 `u64` lanes.
}

simd_f_ty! {
    f64x8: 8, f64, b8x8, f64x8_tests, test_v512 |
    f64, f64, f64, f64, f64, f64, f64, f64 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 512-bit vector with 8 `f64` lanes.
}

impl_from_bits!(
    i8x64: i8,
    i8x64_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    u8x64,
    b8x64
);
impl_from_bits!(
    u8x64: u8,
    u8x64_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    b8x64
);
impl_from_bits!(
    i16x32: i16,
    i16x32_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    u16x32: u16,
    u16x32_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    i32x16: i32,
    i32x16_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    u32x16: u32,
    u32x16_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    f32x16: f32,
    f32x16_from_bits,
    test_v512 | u64x8,
    i64x8,
    f64x8,
    u32x16,
    i32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    i64x8: i64,
    i64x8_from_bits,
    test_v512 | u64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    u64x8: u64,
    u64x8_from_bits,
    test_v512 | i64x8,
    f64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);
impl_from_bits!(
    f64x8: f64,
    f64x8_from_bits,
    test_v512 | u64x8,
    i64x8,
    u32x16,
    i32x16,
    f32x16,
    u16x32,
    i16x32,
    i8x64,
    u8x64,
    b8x64
);

impl_from!(
    f64x8: f64,
    f64x8_from,
    test_v512 | u64x8,
    i64x8,
    u32x8,
    i32x8,
    f32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);
impl_from!(
    i64x8: i64,
    i64x8_from,
    test_v512 | u64x8,
    f64x8,
    u32x8,
    i32x8,
    f32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);
impl_from!(
    u64x8: u64,
    u64x8_from,
    test_v512 | i64x8,
    f64x8,
    u32x8,
    i32x8,
    f32x8,
    u16x8,
    i16x8,
    u8x8,
    i8x8
);

impl_from!(
    f32x16: f32,
    f32x16_from,
    test_v512 | u32x16,
    i32x16,
    u16x16,
    i16x16,
    u8x16,
    i8x16
);
impl_from!(
    i32x16: i32,
    i32x16_from,
    test_v512 | u32x16,
    f32x16,
    u16x16,
    i16x16,
    u8x16,
    i8x16
);
impl_from!(
    u32x16: u32,
    u32x16_from,
    test_v512 | i32x16,
    f32x16,
    u16x16,
    i16x16,
    u8x16,
    i8x16
);

impl_from!(
    i16x32: i16,
    i16x32_from,
    test_v512 | u16x32,
    u8x32,
    i8x32
);
impl_from!(
    u16x32: u16,
    u16x32_from,
    test_v512 | i16x32,
    u8x32,
    i8x32
);

impl_from!(i8x64: i8, i8x64_from, test_v512 | u8x64);
impl_from!(u8x64: u8, u8x64_from, test_v512 | i8x64);
