// Shift operations uniquely typically only have a scalar on the right-hand side.
// Here, we implement shifts for scalar RHS arguments.

use crate::simd::{LaneCount, Simd, SupportedLaneCount};

macro_rules! impl_splatted_shifts {
    { impl $trait:ident :: $trait_fn:ident for $ty:ty } => {
        impl<const N: usize> core::ops::$trait<$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn $trait_fn(self, rhs: $ty) -> Self::Output {
                self.$trait_fn(Simd::splat(rhs))
            }
        }

        impl<const N: usize> core::ops::$trait<&$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            fn $trait_fn(self, rhs: &$ty) -> Self::Output {
                self.$trait_fn(Simd::splat(*rhs))
            }
        }

        impl<'lhs, const N: usize> core::ops::$trait<$ty> for &'lhs Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Simd<$ty, N>;
            #[inline]
            fn $trait_fn(self, rhs: $ty) -> Self::Output {
                self.$trait_fn(Simd::splat(rhs))
            }
        }

        impl<'lhs, const N: usize> core::ops::$trait<&$ty> for &'lhs Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Simd<$ty, N>;
            #[inline]
            fn $trait_fn(self, rhs: &$ty) -> Self::Output {
                self.$trait_fn(Simd::splat(*rhs))
            }
        }
    };
    { $($ty:ty),* } => {
        $(
        impl_splatted_shifts! { impl Shl::shl for $ty }
        impl_splatted_shifts! { impl Shr::shr for $ty }
        )*
    }
}

// In the past there were inference issues when generically splatting arguments.
// Enumerate them instead.
impl_splatted_shifts! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize }
