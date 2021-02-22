#![allow(non_camel_case_types)]

/// Implements inherent methods for a float vector `$name` containing multiple
/// `$lanes` of float `$type`, which uses `$bits_ty` as its binary
/// representation. Called from `define_float_vector!`.
macro_rules! impl_float_vector {
    { $name:ident, $type:ty, $bits_ty:ident } => {
        impl_vector! { $name, $type }

        impl<const LANES: usize> $name<LANES>
        where
            Self: crate::LanesAtMost64,
            crate::$bits_ty<LANES>: crate::LanesAtMost64,
        {
            /// Raw transmutation to an unsigned integer vector type with the
            /// same size and number of lanes.
            #[inline]
            pub fn to_bits(self) -> crate::$bits_ty<LANES> {
                assert_eq!(core::mem::size_of::<Self>(), core::mem::size_of::<crate::$bits_ty<LANES>>());
                unsafe { core::mem::transmute_copy(&self) }
            }

            /// Raw transmutation from an unsigned integer vector type with the
            /// same size and number of lanes.
            #[inline]
            pub fn from_bits(bits: crate::$bits_ty<LANES>) -> Self {
                assert_eq!(core::mem::size_of::<Self>(), core::mem::size_of::<crate::$bits_ty<LANES>>());
                unsafe { core::mem::transmute_copy(&bits) }
            }

            /// Produces a vector where every lane has the absolute value of the
            /// equivalently-indexed lane in `self`.
            #[inline]
            pub fn abs(self) -> Self {
                let no_sign = crate::$bits_ty::splat(!0 >> 1);
                Self::from_bits(self.to_bits() & no_sign)
            }
        }
    };
}


/// A SIMD vector of containing `LANES` `f32` values.
#[repr(simd)]
pub struct SimdF32<const LANES: usize>([f32; LANES])
where
    Self: crate::LanesAtMost64;

impl_float_vector! { SimdF32, f32, SimdU32 }

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }

/// A SIMD vector of containing `LANES` `f64` values.
#[repr(simd)]
pub struct SimdF64<const LANES: usize>([f64; LANES])
where
    Self: crate::LanesAtMost64;

impl_float_vector! { SimdF64, f64, SimdU64 }

from_transmute_x86! { unsafe f64x2 => __m128d }
from_transmute_x86! { unsafe f64x4 => __m256d }
//from_transmute_x86! { unsafe f64x8 => __m512d }

/// Vector of two `f32` values
pub type f32x2 = SimdF32<2>;

/// Vector of four `f32` values
pub type f32x4 = SimdF32<4>;

/// Vector of eight `f32` values
pub type f32x8 = SimdF32<8>;

/// Vector of 16 `f32` values
pub type f32x16 = SimdF32<16>;

/// Vector of two `f64` values
pub type f64x2 = SimdF64<2>;

/// Vector of four `f64` values
pub type f64x4 = SimdF64<4>;

/// Vector of eight `f64` values
pub type f64x8 = SimdF64<8>;
