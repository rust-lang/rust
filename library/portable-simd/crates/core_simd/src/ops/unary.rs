use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::ops::{Neg, Not}; // unary ops

macro_rules! neg {
    ($(impl<const N: usize> Neg for Simd<$scalar:ty, N>)*) => {
        $(impl<const N: usize> Neg for Simd<$scalar, N>
        where
            $scalar: SimdElement,
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                // Safety: `self` is a signed vector
                unsafe { core::intrinsics::simd::simd_neg(self) }
            }
        })*
    }
}

neg! {
    impl<const N: usize> Neg for Simd<f32, N>

    impl<const N: usize> Neg for Simd<f64, N>

    impl<const N: usize> Neg for Simd<i8, N>

    impl<const N: usize> Neg for Simd<i16, N>

    impl<const N: usize> Neg for Simd<i32, N>

    impl<const N: usize> Neg for Simd<i64, N>

    impl<const N: usize> Neg for Simd<isize, N>
}

macro_rules! not {
    ($(impl<const N: usize> Not for Simd<$scalar:ty, N>)*) => {
        $(impl<const N: usize> Not for Simd<$scalar, N>
        where
            $scalar: SimdElement,
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            fn not(self) -> Self::Output {
                self ^ (Simd::splat(!(0 as $scalar)))
            }
        })*
    }
}

not! {
    impl<const N: usize> Not for Simd<i8, N>

    impl<const N: usize> Not for Simd<i16, N>

    impl<const N: usize> Not for Simd<i32, N>

    impl<const N: usize> Not for Simd<i64, N>

    impl<const N: usize> Not for Simd<isize, N>

    impl<const N: usize> Not for Simd<u8, N>

    impl<const N: usize> Not for Simd<u16, N>

    impl<const N: usize> Not for Simd<u32, N>

    impl<const N: usize> Not for Simd<u64, N>

    impl<const N: usize> Not for Simd<usize, N>
}
