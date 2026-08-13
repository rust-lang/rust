#![allow(unused)]
use crate::simd::*;

#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core::arch::aarch64::*;

/// Transmute between `Simd` and ACLE tuple types.
macro_rules! tuple {
    ($scalar:ty,$tuple:ty) => {
        from_transmute! { unsafe Simd<$scalar, { size_of::<$tuple>() / size_of::<$scalar>() }> => $tuple }
    };
    ($scalar:ty,$($tuples:ty),*) => {
        $(tuple! { $scalar, $tuples })*
    };
}

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
    from_transmute! { unsafe f16x4 => float16x4_t }
    from_transmute! { unsafe f16x8 => float16x8_t }

    from_transmute! { unsafe u32x2 => uint32x2_t }
    from_transmute! { unsafe u32x4 => uint32x4_t }
    from_transmute! { unsafe i32x2 => int32x2_t }
    from_transmute! { unsafe i32x4 => int32x4_t }
    from_transmute! { unsafe f32x2 => float32x2_t }
    from_transmute! { unsafe f32x4 => float32x4_t }

    from_transmute! { unsafe Simd<u64, 1> => uint64x1_t }
    from_transmute! { unsafe u64x2 => uint64x2_t }
    from_transmute! { unsafe Simd<i64, 1> => int64x1_t }
    from_transmute! { unsafe i64x2 => int64x2_t }
    from_transmute! { unsafe Simd<u64, 1> => poly64x1_t }
    from_transmute! { unsafe u64x2 => poly64x2_t }

    tuple!(i8, int8x8x2_t, int8x8x3_t, int8x8x4_t);
    tuple!(i8, int8x16x2_t, int8x16x3_t, int8x16x4_t);
    tuple!(u8, uint8x8x2_t, uint8x8x3_t, uint8x8x4_t);
    tuple!(u8, uint8x16x2_t, uint8x16x3_t, uint8x16x4_t);
    tuple!(u8, poly8x8x2_t, poly8x8x3_t, poly8x8x4_t);
    tuple!(u8, poly8x16x2_t, poly8x16x3_t, poly8x16x4_t);

    tuple!(i16, int16x4x2_t, int16x4x3_t, int16x4x4_t);
    tuple!(i16, int16x8x2_t, int16x8x3_t, int16x8x4_t);
    tuple!(u16, uint16x4x2_t, uint16x4x3_t, uint16x4x4_t);
    tuple!(u16, uint16x8x2_t, uint16x8x3_t, uint16x8x4_t);
    tuple!(u16, poly16x4x2_t, poly16x4x3_t, poly16x4x4_t);
    tuple!(u16, poly16x8x2_t, poly16x8x3_t, poly16x8x4_t);
    tuple!(f16, float16x4x2_t, float16x4x3_t, float16x4x4_t);
    tuple!(f16, float16x8x2_t, float16x8x3_t, float16x8x4_t);

    tuple!(i32, int32x2x2_t, int32x2x3_t, int32x2x4_t);
    tuple!(i32, int32x4x2_t, int32x4x3_t, int32x4x4_t);
    tuple!(u32, uint32x2x2_t, uint32x2x3_t, uint32x2x4_t);
    tuple!(u32, uint32x4x2_t, uint32x4x3_t, uint32x4x4_t);
    tuple!(f32, float32x2x2_t, float32x2x3_t, float32x2x4_t);
    tuple!(f32, float32x4x2_t, float32x4x3_t, float32x4x4_t);

    tuple!(i64, int64x1x2_t, int64x1x3_t, int64x1x4_t);
    tuple!(i64, int64x2x2_t, int64x2x3_t, int64x2x4_t);
    tuple!(u64, uint64x1x2_t, uint64x1x3_t, uint64x1x4_t);
    tuple!(u64, uint64x2x2_t, uint64x2x3_t, uint64x2x4_t);
    tuple!(u64, poly64x1x2_t, poly64x1x3_t, poly64x1x4_t);
    tuple!(u64, poly64x2x2_t, poly64x2x3_t, poly64x2x4_t);
}

#[cfg(any(
    all(target_feature = "v6", not(target_feature = "mclass")),
    all(target_feature = "mclass", target_feature = "dsp"),
))]
mod simd32 {
    use super::*;

    from_transmute! { unsafe Simd<u8, 4> => uint8x4_t }
    from_transmute! { unsafe Simd<i8, 4> => int8x4_t }
    from_transmute! { unsafe Simd<u16, 2> => uint16x2_t }
    from_transmute! { unsafe Simd<i16, 2> => int16x2_t }
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

    tuple!(f64, float64x1x2_t, float64x1x3_t, float64x1x4_t);
    tuple!(f64, float64x2x2_t, float64x2x3_t, float64x2x4_t);
}
