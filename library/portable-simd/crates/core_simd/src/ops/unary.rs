use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::ops::{Neg, Not}; // unary ops

macro_rules! neg {
    ($(impl<const LANES: usize> Neg for Simd<$scalar:ty, LANES>)*) => {
        $(impl<const LANES: usize> Neg for Simd<$scalar, LANES>
        where
            $scalar: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the input"]
            fn neg(self) -> Self::Output {
                unsafe { intrinsics::simd_neg(self) }
            }
        })*
    }
}

neg! {
    impl<const LANES: usize> Neg for Simd<f32, LANES>

    impl<const LANES: usize> Neg for Simd<f64, LANES>

    impl<const LANES: usize> Neg for Simd<i8, LANES>

    impl<const LANES: usize> Neg for Simd<i16, LANES>

    impl<const LANES: usize> Neg for Simd<i32, LANES>

    impl<const LANES: usize> Neg for Simd<i64, LANES>

    impl<const LANES: usize> Neg for Simd<isize, LANES>
}

macro_rules! not {
    ($(impl<const LANES: usize> Not for Simd<$scalar:ty, LANES>)*) => {
        $(impl<const LANES: usize> Not for Simd<$scalar, LANES>
        where
            $scalar: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the input"]
            fn not(self) -> Self::Output {
                self ^ (Simd::splat(!(0 as $scalar)))
            }
        })*
    }
}

not! {
    impl<const LANES: usize> Not for Simd<i8, LANES>

    impl<const LANES: usize> Not for Simd<i16, LANES>

    impl<const LANES: usize> Not for Simd<i32, LANES>

    impl<const LANES: usize> Not for Simd<i64, LANES>

    impl<const LANES: usize> Not for Simd<isize, LANES>

    impl<const LANES: usize> Not for Simd<u8, LANES>

    impl<const LANES: usize> Not for Simd<u16, LANES>

    impl<const LANES: usize> Not for Simd<u32, LANES>

    impl<const LANES: usize> Not for Simd<u64, LANES>

    impl<const LANES: usize> Not for Simd<usize, LANES>
}
