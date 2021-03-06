#![allow(non_camel_case_types)]

/// Implements inherent methods for a float vector `$name` containing multiple
/// `$lanes` of float `$type`, which uses `$bits_ty` as its binary
/// representation. Called from `define_float_vector!`.
macro_rules! impl_float_vector {
    { $name:ident, $type:ty, $bits_ty:ident, $mask_ty:ident, $mask_impl_ty:ident } => {
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

        impl<const LANES: usize> $name<LANES>
        where
            Self: crate::LanesAtMost64,
            crate::$bits_ty<LANES>: crate::LanesAtMost64,
            crate::$mask_impl_ty<LANES>: crate::LanesAtMost64,
        {
            /// Returns true for each lane if it has a positive sign, including
            /// `+0.0`, `NaN`s with positive sign bit and positive infinity.
            #[inline]
            pub fn is_sign_positive(self) -> crate::$mask_ty<LANES> {
                !self.is_sign_negative()
            }

            /// Returns true for each lane if it has a negative sign, including
            /// `-0.0`, `NaN`s with negative sign bit and negative infinity.
            #[inline]
            pub fn is_sign_negative(self) -> crate::$mask_ty<LANES> {
                let sign_bits = self.to_bits() & crate::$bits_ty::splat((!0 >> 1) + 1);
                sign_bits.lanes_gt(crate::$bits_ty::splat(0))
            }

            /// Returns true for each lane if its value is `NaN`.
            #[inline]
            pub fn is_nan(self) -> crate::$mask_ty<LANES> {
                self.lanes_ne(self)
            }

            /// Returns true for each lane if its value is positive infinity or negative infinity.
            #[inline]
            pub fn is_infinite(self) -> crate::$mask_ty<LANES> {
                self.abs().lanes_eq(Self::splat(<$type>::INFINITY))
            }

            /// Returns true for each lane if its value is neither infinite nor `NaN`.
            #[inline]
            pub fn is_finite(self) -> crate::$mask_ty<LANES> {
                self.abs().lanes_lt(Self::splat(<$type>::INFINITY))
            }

            /// Returns true for each lane if its value is subnormal.
            #[inline]
            pub fn is_subnormal(self) -> crate::$mask_ty<LANES> {
                self.abs().lanes_ne(Self::splat(0.0)) & (self.to_bits() & Self::splat(<$type>::INFINITY).to_bits()).lanes_eq(crate::$bits_ty::splat(0))
            }

            /// Returns true for each lane if its value is neither neither zero, infinite,
            /// subnormal, or `NaN`.
            #[inline]
            pub fn is_normal(self) -> crate::$mask_ty<LANES> {
                !(self.abs().lanes_eq(Self::splat(0.0)) | self.is_nan() | self.is_subnormal() | self.is_infinite())
            }
        }
    };
}

/// A SIMD vector of containing `LANES` `f32` values.
#[repr(simd)]
pub struct SimdF32<const LANES: usize>([f32; LANES])
where
    Self: crate::LanesAtMost64;

impl_float_vector! { SimdF32, f32, SimdU32, Mask32, SimdI32 }

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }

/// A SIMD vector of containing `LANES` `f64` values.
#[repr(simd)]
pub struct SimdF64<const LANES: usize>([f64; LANES])
where
    Self: crate::LanesAtMost64;

impl_float_vector! { SimdF64, f64, SimdU64, Mask64, SimdI64 }

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
