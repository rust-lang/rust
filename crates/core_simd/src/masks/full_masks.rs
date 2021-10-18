//! Masks that take up full SIMD vector registers.

use super::MaskElement;
use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SupportedLaneCount};

#[repr(transparent)]
pub struct Mask<T, const LANES: usize>(Simd<T, LANES>)
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount;

impl<T, const LANES: usize> Copy for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Clone for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const LANES: usize> PartialEq for Mask<T, LANES>
where
    T: MaskElement + PartialEq,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const LANES: usize> PartialOrd for Mask<T, LANES>
where
    T: MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const LANES: usize> Eq for Mask<T, LANES>
where
    T: MaskElement + Eq,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Ord for Mask<T, LANES>
where
    T: MaskElement + Ord,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T, const LANES: usize> Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn splat(value: bool) -> Self {
        Self(Simd::splat(if value { T::TRUE } else { T::FALSE }))
    }

    #[inline]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        T::eq(self.0[lane], T::TRUE)
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0[lane] = if value { T::TRUE } else { T::FALSE }
    }

    #[inline]
    pub fn to_int(self) -> Simd<T, LANES> {
        self.0
    }

    #[inline]
    pub unsafe fn from_int_unchecked(value: Simd<T, LANES>) -> Self {
        Self(value)
    }

    #[inline]
    pub fn convert<U>(self) -> Mask<U, LANES>
    where
        U: MaskElement,
    {
        unsafe { Mask(intrinsics::simd_cast(self.0)) }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        unsafe {
            // TODO remove the transmute when rustc can use arrays of u8 as bitmasks
            assert_eq!(
                core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::IntBitMask>(),
                LaneCount::<LANES>::BITMASK_LEN,
            );
            let bitmask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                intrinsics::simd_bitmask(self.0);
            let mut bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN] =
                core::mem::transmute_copy(&bitmask);

            // There is a bug where LLVM appears to implement this operation with the wrong
            // bit order.
            // TODO fix this in a better way
            if cfg!(target_endian = "big") {
                for x in bitmask.as_mut() {
                    *x = x.reverse_bits();
                }
            }

            bitmask
        }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    pub fn from_bitmask(mut bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        unsafe {
            // There is a bug where LLVM appears to implement this operation with the wrong
            // bit order.
            // TODO fix this in a better way
            if cfg!(target_endian = "big") {
                for x in bitmask.as_mut() {
                    *x = x.reverse_bits();
                }
            }

            // TODO remove the transmute when rustc can use arrays of u8 as bitmasks
            assert_eq!(
                core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::IntBitMask>(),
                LaneCount::<LANES>::BITMASK_LEN,
            );
            let bitmask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                core::mem::transmute_copy(&bitmask);

            Self::from_int_unchecked(intrinsics::simd_select_bitmask(
                bitmask,
                Self::splat(true).to_int(),
                Self::splat(false).to_int(),
            ))
        }
    }

    #[inline]
    pub fn any(self) -> bool {
        unsafe { intrinsics::simd_reduce_any(self.to_int()) }
    }

    #[inline]
    pub fn all(self) -> bool {
        unsafe { intrinsics::simd_reduce_all(self.to_int()) }
    }
}

impl<T, const LANES: usize> core::convert::From<Mask<T, LANES>> for Simd<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from(value: Mask<T, LANES>) -> Self {
        value.0
    }
}

impl<T, const LANES: usize> core::ops::BitAnd for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(intrinsics::simd_and(self.0, rhs.0)) }
    }
}

impl<T, const LANES: usize> core::ops::BitOr for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(intrinsics::simd_or(self.0, rhs.0)) }
    }
}

impl<T, const LANES: usize> core::ops::BitXor for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { Self(intrinsics::simd_xor(self.0, rhs.0)) }
    }
}

impl<T, const LANES: usize> core::ops::Not for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        Self::splat(true) ^ self
    }
}
