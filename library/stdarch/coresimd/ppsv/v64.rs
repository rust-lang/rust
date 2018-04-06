//! 64-bit wide portable packed vector types.

simd_i_ty! {
    i8x8: 8, i8, m8x8, i8x8_tests, test_v64 |
    i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector with 8 `i8` lanes.
}

simd_u_ty! {
    u8x8: 8, u8, m8x8, u8x8_tests, test_v64 |
    u8, u8, u8, u8, u8, u8, u8, u8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector with 8 `u8` lanes.
}

simd_m_ty! {
    m8x8: 8, i8, m8x8_tests, test_v64 |
    i8, i8, i8, i8, i8, i8, i8, i8 |
    x0, x1, x2, x3, x4, x5, x6, x7 |
    /// A 64-bit vector mask with 8 lanes.
}

simd_i_ty! {
    i16x4: 4, i16, m16x4, i16x4_tests, test_v64 |
    i16, i16, i16, i16 |
    x0, x1, x2, x3 |
    /// A 64-bit vector with 4 `i16` lanes.
}

simd_u_ty! {
    u16x4: 4, u16, m16x4, u16x4_tests, test_v64 |
    u16, u16, u16, u16 |
    x0, x1, x2, x3 |
    /// A 64-bit vector with 4 `u16` lanes.
}

simd_m_ty! {
    m16x4: 4, i16, m16x4_tests, test_v64 |
    i16, i16, i16, i16 |
    x0, x1, x2, x3 |
    /// A 64-bit vector mask with 4 lanes.
}

simd_i_ty! {
    i32x2: 2, i32, m32x2, i32x2_tests, test_v64 |
    i32, i32 |
    x0, x1 |
    /// A 64-bit vector with 2 `i32` lanes.
}

simd_u_ty! {
    u32x2: 2, u32, m32x2, u32x2_tests, test_v64 |
    u32, u32 |
    x0, x1 |
    /// A 64-bit vector with 2 `u32` lanes.
}

simd_m_ty! {
    m32x2: 2, i32, m32x2_tests, test_v64 |
    i32, i32 |
    x0, x1 |
    /// A 64-bit vector mask with 2 lanes.
}

simd_f_ty! {
    f32x2: 2, f32, m32x2, f32x2_tests, test_v64 |
    f32, f32 |
    x0, x1 |
    /// A 64-bit vector with 2 `f32` lanes.
}

#[cfg(target_arch = "x86")]
use coresimd::arch::x86::__m64;

#[cfg(target_arch = "x86_64")]
use coresimd::arch::x86_64::__m64;

macro_rules! from_bits_x86 {
    ($id:ident, $elem_ty:ident, $test_mod:ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl_from_bits_!($id: __m64);
    };
}

#[cfg(all(target_arch = "arm", target_feature = "neon",
          target_feature = "v7"))]
use coresimd::arch::arm::{// FIXME: float16x4_t,
                          float32x2_t,
                          int16x4_t,
                          int32x2_t,
                          int64x1_t,
                          int8x8_t,
                          poly16x4_t,
                          poly8x8_t,
                          uint16x4_t,
                          uint32x2_t,
                          uint64x1_t,
                          uint8x8_t};

#[cfg(target_arch = "aarch64")]
use coresimd::arch::aarch64::{// FIXME: float16x4_t,
                              float32x2_t,
                              float64x1_t,
                              int16x4_t,
                              int32x2_t,
                              int64x1_t,
                              int8x8_t,
                              poly16x4_t,
                              poly8x8_t,
                              uint16x4_t,
                              uint32x2_t,
                              uint64x1_t,
                              uint8x8_t};

macro_rules! from_bits_arm {
    ($id:ident, $elem_ty:ident, $test_mod_arm:ident, $test_mod_a64:ident) => {
        #[cfg(any(all(target_arch = "arm", target_feature = "neon",
                      target_feature = "v7"),
                  target_arch = "aarch64"))]
        impl_from_bits_!(
            $id: int64x1_t,
            uint64x1_t,
            uint32x2_t,
            int32x2_t,
            float32x2_t,
            uint16x4_t,
            int16x4_t,
            // FIXME: float16x4_t
            poly16x4_t,
            uint8x8_t,
            int8x8_t,
            poly8x8_t
        );
        #[cfg(target_arch = "aarch64")]
        impl_from_bits_!($id: float64x1_t);
    };
}

impl_from_bits!(
    u32x2: u32,
    u32x2_from_bits,
    test_v64 | i32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
from_bits_x86!(u32x2, u32, u32x2_from_bits_x86);
from_bits_arm!(
    u32x2,
    u32,
    u32x2_from_bits_arm,
    u32x2_from_bits_aarch64
);

impl_from_bits!(
    i32x2: i32,
    i32x2_from_bits,
    test_v64 | u32x2,
    f32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
from_bits_x86!(i32x2, i32, i32x2_from_bits_x86);
from_bits_arm!(
    i32x2,
    i32,
    i32x2_from_bits_arm,
    i32x2_from_bits_aarch64
);

impl_from_bits!(
    f32x2: f32,
    f32x2_from_bits,
    test_v64 | i32x2,
    u32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
from_bits_x86!(f32x2, f32, f32x2_from_bits_x86);
from_bits_arm!(
    f32x2,
    f32,
    f32x2_from_bits_arm,
    f32x2_from_bits_aarch64
);

impl_from_bits!(
    u16x4: u16,
    u16x4_from_bits,
    test_v64 | u32x2,
    i32x2,
    m32x2,
    i16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
from_bits_x86!(u16x4, u16, u16x4_from_bits_x86);
from_bits_arm!(
    u16x4,
    u16,
    u16x4_from_bits_arm,
    u16x4_from_bits_aarch64
);

impl_from_bits!(
    i16x4: i16,
    i16x4_from_bits,
    test_v64 | u32x2,
    i32x2,
    m32x2,
    u16x4,
    m16x4,
    u8x8,
    i8x8,
    m8x8
);
from_bits_x86!(i16x4, i16, i16x4_from_bits_x86);
from_bits_arm!(
    i16x4,
    i16,
    i16x4_from_bits_arm,
    i16x4_from_bits_aarch64
);

impl_from_bits!(
    u8x8: u8,
    u8x8_from_bits,
    test_v64 | u32x2,
    i32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    i8x8,
    m8x8
);
from_bits_x86!(u8x8, u8, u8x8_from_bits_x86);
from_bits_arm!(
    u8x8,
    u8,
    u8x8_from_bits_arm,
    u8x8_from_bits_aarch64
);

impl_from_bits!(
    i8x8: i8,
    i8x8_from_bits,
    test_v64 | u32x2,
    i32x2,
    m32x2,
    u16x4,
    i16x4,
    m16x4,
    u8x8,
    m8x8
);
from_bits_x86!(i8x8, i8, i8x8_from_bits_x86);
from_bits_arm!(
    i8x8,
    i8,
    i8x8_from_bits_arm,
    i8x8_from_bits_aarch64
);

impl_from!(
    f32x2: f32,
    f32x2_from,
    test_v64 | f64x2,
    u64x2,
    i64x2,
    m64x2,
    u32x2,
    i32x2,
    m32x2,
    u16x2,
    i16x2,
    m16x2,
    u8x2,
    i8x2,
    m8x2
);

impl_from!(
    u32x2: u32,
    u32x2_from,
    test_v64 | f64x2,
    u64x2,
    i64x2,
    m64x2,
    f32x2,
    i32x2,
    m32x2,
    u16x2,
    i16x2,
    m16x2,
    u8x2,
    i8x2,
    m8x2
);

impl_from!(
    i32x2: i32,
    i32x2_from,
    test_v64 | f64x2,
    u64x2,
    i64x2,
    m64x2,
    f32x2,
    u32x2,
    m32x2,
    u16x2,
    i16x2,
    m16x2,
    u8x2,
    i8x2,
    m8x2
);

impl_from!(
    u16x4: u16,
    u16x4_from,
    test_v64 | f64x4,
    u64x4,
    i64x4,
    m64x4,
    f32x4,
    i32x4,
    u32x4,
    m32x4,
    i16x4,
    m16x4,
    u8x4,
    i8x4,
    m8x4
);

impl_from!(
    i16x4: i16,
    i16x4_from,
    test_v64 | f64x4,
    u64x4,
    i64x4,
    m64x4,
    f32x4,
    i32x4,
    u32x4,
    m32x4,
    u16x4,
    m16x4,
    u8x4,
    i8x4,
    m8x4
);
impl_from!(
    i8x8: i8,
    i8x8_from,
    test_v64 | f64x8,
    u64x8,
    i64x8,
    m1x8,
    f32x8,
    u32x8,
    i32x8,
    m32x8,
    i16x8,
    u16x8,
    m16x8,
    u8x8,
    m8x8
);
impl_from!(
    u8x8: u8,
    u8x8_from,
    test_v64 | f64x8,
    u64x8,
    i64x8,
    m1x8,
    f32x8,
    u32x8,
    i32x8,
    m32x8,
    i16x8,
    u16x8,
    m16x8,
    i8x8,
    m8x8
);

impl_from!(
    m8x8: i8,
    m8x8_from,
    test_v64 | m1x8,
    m32x8,
    m16x8
);

impl_from!(
    m16x4: i16,
    m16x4_from,
    test_v64 | m64x4,
    m32x4,
    m8x4
);

impl_from!(
    m32x2: i32,
    m32x2_from,
    test_v64 | m64x2,
    m16x2,
    m8x2
);
