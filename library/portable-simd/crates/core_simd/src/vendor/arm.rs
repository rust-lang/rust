#![allow(unused)]
use crate::simd::*;

#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core::arch::aarch64::*;

#[cfg(all(
    any(
        target_arch = "aarch64",
        target_arch = "arm64ec",
        all(target_arch = "arm", target_feature = "v7"),
    ),
    target_endian = "little"
))]
mod neon {
    use super::*;

    from_transmute! { unsafe f32x2 => float32x2_t }
    from_transmute! { unsafe f32x4 => float32x4_t }

    from_transmute! { unsafe u8x8 => uint8x8_t }
    from_transmute! { unsafe u8x16 => uint8x16_t }
    from_transmute! { unsafe i8x8 => int8x8_t }
    from_transmute! { unsafe i8x16 => int8x16_t }
    from_transmute! { unsafe u8x8 => poly8x8_t }
    from_transmute! { unsafe u8x16 => poly8x16_t }

    from_transmute! { unsafe u16x4 => uint16x4_t }
    from_transmute! { unsafe u16x8 => uint16x8_t }
    from_transmute! { unsafe i16x4 => int16x4_t }
    from_transmute! { unsafe i16x8 => int16x8_t }
    from_transmute! { unsafe u16x4 => poly16x4_t }
    from_transmute! { unsafe u16x8 => poly16x8_t }

    from_transmute! { unsafe u32x2 => uint32x2_t }
    from_transmute! { unsafe u32x4 => uint32x4_t }
    from_transmute! { unsafe i32x2 => int32x2_t }
    from_transmute! { unsafe i32x4 => int32x4_t }

    from_transmute! { unsafe Simd<u64, 1> => uint64x1_t }
    from_transmute! { unsafe u64x2 => uint64x2_t }
    from_transmute! { unsafe Simd<i64, 1> => int64x1_t }
    from_transmute! { unsafe i64x2 => int64x2_t }
    from_transmute! { unsafe Simd<u64, 1> => poly64x1_t }
    from_transmute! { unsafe u64x2 => poly64x2_t }
}

#[cfg(any(
    all(target_feature = "v5te", not(target_feature = "mclass")),
    all(target_feature = "mclass", target_feature = "dsp"),
))]
mod dsp {
    use super::*;

    from_transmute! { unsafe Simd<u16, 2> => uint16x2_t }
    from_transmute! { unsafe Simd<i16, 2> => int16x2_t }
}

#[cfg(any(
    all(target_feature = "v6", not(target_feature = "mclass")),
    all(target_feature = "mclass", target_feature = "dsp"),
))]
mod simd32 {
    use super::*;

    from_transmute! { unsafe Simd<u8, 4> => uint8x4_t }
    from_transmute! { unsafe Simd<i8, 4> => int8x4_t }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm64ec"),
    target_endian = "little"
))]
mod aarch64 {
    use super::neon::*;
    use super::*;

    from_transmute! { unsafe Simd<f64, 1> => float64x1_t }
    from_transmute! { unsafe f64x2 => float64x2_t }
}
