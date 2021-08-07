//! Masks that take up full SIMD vector registers.

use super::MaskElement;
use crate::{LaneCount, Simd, SupportedLaneCount};

#[repr(transparent)]
pub struct Mask<Element, const LANES: usize>(Simd<Element, LANES>)
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount;

impl<Element, const LANES: usize> Copy for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<Element, const LANES: usize> Clone for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Element, const LANES: usize> PartialEq for Mask<Element, LANES>
where
    Element: MaskElement + PartialEq,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<Element, const LANES: usize> PartialOrd for Mask<Element, LANES>
where
    Element: MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<Element, const LANES: usize> Eq for Mask<Element, LANES>
where
    Element: MaskElement + Eq,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<Element, const LANES: usize> Ord for Mask<Element, LANES>
where
    Element: MaskElement + Ord,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<Element, const LANES: usize> Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn splat(value: bool) -> Self {
        Self(Simd::splat(if value {
            Element::TRUE
        } else {
            Element::FALSE
        }))
    }

    #[inline]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        Element::eq(self.0[lane], Element::TRUE)
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0[lane] = if value { Element::TRUE } else { Element::FALSE }
    }

    #[inline]
    pub fn to_int(self) -> Simd<Element, LANES> {
        self.0
    }

    #[inline]
    pub unsafe fn from_int_unchecked(value: Simd<Element, LANES>) -> Self {
        Self(value)
    }

    #[inline]
    pub fn convert<T>(self) -> Mask<T, LANES>
    where
        T: MaskElement,
    {
        unsafe { Mask(crate::intrinsics::simd_cast(self.0)) }
    }

    #[inline]
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        unsafe {
            // TODO remove the transmute when rustc can use arrays of u8 as bitmasks
            assert_eq!(
                core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::IntBitMask>(),
                LaneCount::<LANES>::BITMASK_LEN,
            );
            let bitmask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                crate::intrinsics::simd_bitmask(self.0);
            let mut bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN] =
                core::mem::transmute_copy(&bitmask);

            // There is a bug where LLVM appears to implement this operation with the wrong
            // bit order.
            // TODO fix this in a better way
            if cfg!(any(target_arch = "mips", target_arch = "mips64")) {
                for x in bitmask.as_mut() {
                    *x = x.reverse_bits();
                }
            }

            bitmask
        }
    }

    #[inline]
    pub fn from_bitmask(mut bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        unsafe {
            // There is a bug where LLVM appears to implement this operation with the wrong
            // bit order.
            // TODO fix this in a better way
            if cfg!(any(target_arch = "mips", target_arch = "mips64")) {
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

            Self::from_int_unchecked(crate::intrinsics::simd_select_bitmask(
                bitmask,
                Self::splat(true).to_int(),
                Self::splat(false).to_int(),
            ))
        }
    }
}

impl<Element, const LANES: usize> core::convert::From<Mask<Element, LANES>> for Simd<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from(value: Mask<Element, LANES>) -> Self {
        value.0
    }
}

impl<Element, const LANES: usize> core::ops::BitAnd for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(crate::intrinsics::simd_and(self.0, rhs.0)) }
    }
}

impl<Element, const LANES: usize> core::ops::BitOr for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(crate::intrinsics::simd_or(self.0, rhs.0)) }
    }
}

impl<Element, const LANES: usize> core::ops::BitXor for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { Self(crate::intrinsics::simd_xor(self.0, rhs.0)) }
    }
}

impl<Element, const LANES: usize> core::ops::Not for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        Self::splat(true) ^ self
    }
}

impl_full_mask_reductions! {}
