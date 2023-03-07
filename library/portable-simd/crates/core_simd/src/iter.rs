use crate::simd::{LaneCount, Simd, SupportedLaneCount};
use core::{
    iter::{Product, Sum},
    ops::{Add, Mul},
};

macro_rules! impl_traits {
    { $type:ty } => {
        impl<const LANES: usize> Sum<Self> for Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Simd::splat(0 as $type), Add::add)
            }
        }

        impl<const LANES: usize> Product<Self> for Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Simd::splat(1 as $type), Mul::mul)
            }
        }

        impl<'a, const LANES: usize> Sum<&'a Self> for Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Simd::splat(0 as $type), Add::add)
            }
        }

        impl<'a, const LANES: usize> Product<&'a Self> for Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Simd::splat(1 as $type), Mul::mul)
            }
        }
    }
}

impl_traits! { f32 }
impl_traits! { f64 }
impl_traits! { u8 }
impl_traits! { u16 }
impl_traits! { u32 }
impl_traits! { u64 }
impl_traits! { usize }
impl_traits! { i8 }
impl_traits! { i16 }
impl_traits! { i32 }
impl_traits! { i64 }
impl_traits! { isize }
