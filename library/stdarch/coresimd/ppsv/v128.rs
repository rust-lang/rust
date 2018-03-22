//! 128-bit wide portable packed vector types.
use coresimd::simd::{b8x2, b8x4, b8x8};

simd_i_ty! {
    i8x16: 16, i8, b8x16, i8x16_tests, test_v128 |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8  |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `i8` lanes.
}

simd_u_ty! {
    u8x16: 16, u8, b8x16, u8x16_tests, test_v128 |
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `u8` lanes.
}

simd_b_ty! {
    b8x16: 16, i8, b8x16_tests, test_v128 |
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8  |
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
    /// A 128-bit vector with 16 `bool` lanes.
}

simd_i_ty! {
    i16x8: 8, i16, b8x8, i16x8_tests, test_v128 |
    i16, i16, i16, i16, i16, i16, i16, i16 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 128-bit vector with 8 `i16` lanes.
}

simd_u_ty! {
    u16x8: 8, u16, b8x8, u16x8_tests, test_v128 |
    u16, u16, u16, u16, u16, u16, u16, u16 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 128-bit vector with 8 `u16` lanes.
}

simd_i_ty! {
    i32x4: 4, i32, b8x4, i32x4_tests, test_v128 |
    i32, i32, i32, i32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `i32` lanes.
}

simd_u_ty! {
    u32x4: 4, u32, b8x4, u32x4_tests, test_v128 |
    u32, u32, u32, u32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `u32` lanes.
}

simd_f_ty! {
    f32x4: 4, f32, b8x4, f32x4_tests, test_v128 |
    f32, f32, f32, f32 |
    x0, x1, x2, x3 |
    /// A 128-bit vector with 4 `f32` lanes.
}

simd_i_ty! {
    i64x2: 2, i64, b8x2, i64x2_tests, test_v128 |
    i64, i64 |
    x0, x1 |
    /// A 128-bit vector with 2 `u64` lanes.
}

simd_u_ty! {
    u64x2: 2, u64, b8x2, u64x2_tests, test_v128 |
    u64, u64 |
    x0, x1 |
    /// A 128-bit vector with 2 `u64` lanes.
}

simd_f_ty! {
    f64x2: 2, f64, b8x2, f64x2_tests, test_v128 |
    f64, f64 |
    x0, x1 |
    /// A 128-bit vector with 2 `f64` lanes.
}

#[cfg(target_arch = "x86")]
use coresimd::arch::x86::{__m128, __m128d, __m128i};
#[cfg(target_arch = "x86_64")]
use coresimd::arch::x86_64::{__m128, __m128d, __m128i};

macro_rules! from_bits_x86 {
    ($id: ident, $elem_ty: ident, $test_mod: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl_from_bits_!($id: __m128, __m128i, __m128d);
    };
}

#[cfg(all(target_arch = "arm", target_feature = "v7"))]
use coresimd::arch::arm::{// FIXME: float16x8_t,
                          float32x4_t,
                          int16x8_t,
                          int32x4_t,
                          int64x2_t,
                          int8x16_t,
                          poly16x8_t,
                          poly8x16_t,
                          uint16x8_t,
                          uint32x4_t,
                          uint64x2_t,
                          uint8x16_t};

#[cfg(target_arch = "aarch64")]
use coresimd::arch::aarch64::{// FIXME: float16x8_t,
                              float32x4_t,
                              float64x2_t,
                              int16x8_t,
                              int32x4_t,
                              int64x2_t,
                              int8x16_t,
                              poly16x8_t,
                              poly8x16_t,
                              uint16x8_t,
                              uint32x4_t,
                              uint64x2_t,
                              uint8x16_t};

macro_rules! from_bits_arm {
    ($id:ident, $elem_ty:ident, $test_mod_arm:ident, $test_mod_a64:ident) => {
        #[cfg(any(all(target_arch = "arm", target_feature = "v7"), target_arch = "aarch64"))]
        impl_from_bits_!(
            $id:
            int8x16_t,
            uint8x16_t,
            int16x8_t,
            uint16x8_t,
            int32x4_t,
            uint32x4_t,
            int64x2_t,
            uint64x2_t,
            // FIXME: float16x8_t,
            float32x4_t,
            poly8x16_t,
            poly16x8_t
        );
        #[cfg(target_arch = "aarch64")]
        impl_from_bits_!(
            $id: float64x2_t
        );
    }
}

impl_from_bits!(
    u64x2: u64,
    u64x2_from_bits,
    test_v128 | i64x2,
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
from_bits_x86!(u64x2, u64, u64x2_from_bits_x86);
from_bits_arm!(
    u64x2,
    u64,
    u64x2_from_bits_arm,
    u64x2_from_bits_aarch64
);

impl_from_bits!(
    i64x2: i64,
    i64x2_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(i64x2, i64, i64x2_from_bits_x86);
from_bits_arm!(
    i64x2,
    i64,
    i64x2_from_bits_arm,
    i64x2_from_bits_aarch64
);

impl_from_bits!(
    f64x2: f64,
    f64x2_from_bits,
    test_v128 | i64x2,
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
from_bits_x86!(f64x2, f64, f64x2_from_bits_x86);
from_bits_arm!(
    f64x2,
    f64,
    f64x2_from_bits_arm,
    f64x2_from_bits_aarch64
);

impl_from_bits!(
    u32x4: u32,
    u32x4_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(u32x4, u32, u32x4_from_bits_x86);
from_bits_arm!(
    u32x4,
    u32,
    u32x4_from_bits_arm,
    u32x4_from_bits_aarch64
);

impl_from_bits!(
    i32x4: i32,
    i32x4_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(i32x4, i32, i32x4_from_bits_x86);
from_bits_arm!(
    i32x4,
    i32,
    i32x4_from_bits_arm,
    i32x4_from_bits_aarch64
);

impl_from_bits!(
    f32x4: f32,
    f32x4_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(f32x4, f32, f32x4_from_bits_x86);
from_bits_arm!(
    f32x4,
    f32,
    f32x4_from_bits_arm,
    f32x4_from_bits_aarch64
);

impl_from_bits!(
    u16x8: u16,
    u16x8_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(u16x8, u16, u16x8_from_bits_x86);
from_bits_arm!(
    u16x8,
    u16,
    u16x8_from_bits_arm,
    u16x8_from_bits_aarch64
);

impl_from_bits!(
    i16x8: i16,
    i16x8_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(i16x8, i16, i16x8_from_bits_x86);
from_bits_arm!(
    i16x8,
    i16,
    i16x8_from_bits_arm,
    i16x8_from_bits_aarch64
);

impl_from_bits!(
    u8x16: u8,
    u8x16_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(u8x16, u8, u8x16_from_bits_x86);
from_bits_arm!(
    u8x16,
    u8,
    u8x16_from_bits_arm,
    u8x16_from_bits_aarch64
);

impl_from_bits!(
    i8x16: i8,
    i8x16_from_bits,
    test_v128 | u64x2,
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
from_bits_x86!(i8x16, i8, i8x16_from_bits_x86);
from_bits_arm!(
    i8x16,
    i8,
    i8x16_from_bits_arm,
    i8x16_from_bits_aarch64
);

impl_from!(
    f64x2: f64,
    f64x2_from,
    test_v128 | f32x2,
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
    f32x4_from,
    test_v128 | f64x4,
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
    u64x2_from,
    test_v128 | f32x2,
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
    i64x2_from,
    test_v128 | f32x2,
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
    u32x4_from,
    test_v128 | f64x4,
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
    i32x4_from,
    test_v128 | f64x4,
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
    i16x8_from,
    test_v128 | f64x8,
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
    u16x8_from,
    test_v128 | f64x8,
    u64x8,
    i64x8,
    f32x8,
    u32x8,
    i32x8,
    i16x8,
    u8x8,
    i8x8
);
