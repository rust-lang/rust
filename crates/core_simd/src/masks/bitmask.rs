use crate::{LaneCount, MaskElement, Simd, SupportedLaneCount};
use core::marker::PhantomData;

/// A mask where each lane is represented by a single bit.
#[repr(transparent)]
pub struct Mask<Element, const LANES: usize>(
    <LaneCount<LANES> as SupportedLaneCount>::BitMask,
    PhantomData<Element>,
)
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
    fn clone(&self) -> Self {
        *self
    }
}

impl<Element, const LANES: usize> PartialEq for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<Element, const LANES: usize> PartialOrd for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.as_ref().partial_cmp(other.0.as_ref())
    }
}

impl<Element, const LANES: usize> Eq for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<Element, const LANES: usize> Ord for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl<Element, const LANES: usize> Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    pub fn splat(value: bool) -> Self {
        let mut mask = <LaneCount<LANES> as SupportedLaneCount>::BitMask::default();
        if value {
            mask.as_mut().fill(u8::MAX)
        } else {
            mask.as_mut().fill(u8::MIN)
        }
        if LANES % 8 > 0 {
            *mask.as_mut().last_mut().unwrap() &= u8::MAX >> (8 - LANES % 8);
        }
        Self(mask, PhantomData)
    }

    #[inline]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        (self.0.as_ref()[lane / 8] >> (lane % 8)) & 0x1 > 0
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0.as_mut()[lane / 8] ^= ((value ^ self.test_unchecked(lane)) as u8) << (lane % 8)
    }

    #[inline]
    pub fn to_int(self) -> Simd<Element, LANES> {
        unsafe {
            let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                core::mem::transmute_copy(&self);
            crate::intrinsics::simd_select_bitmask(
                mask,
                Simd::splat(Element::TRUE),
                Simd::splat(Element::FALSE),
            )
        }
    }

    #[inline]
    pub unsafe fn from_int_unchecked(value: Simd<Element, LANES>) -> Self {
        // TODO remove the transmute when rustc is more flexible
        assert_eq!(
            core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::BitMask>(),
            core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::IntBitMask>(),
        );
        let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
            crate::intrinsics::simd_bitmask(value);
        Self(core::mem::transmute_copy(&mask), PhantomData)
    }

    #[inline]
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        // Safety: these are the same type and we are laundering the generic
        unsafe { core::mem::transmute_copy(&self.0) }
    }

    #[inline]
    pub fn from_bitmask(bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        // Safety: these are the same type and we are laundering the generic
        Self(unsafe { core::mem::transmute_copy(&bitmask) }, PhantomData)
    }

    #[inline]
    pub fn convert<T>(self) -> Mask<T, LANES>
    where
        T: MaskElement,
    {
        unsafe { core::mem::transmute_copy(&self) }
    }

    #[inline]
    pub fn any(self) -> bool {
        self != Self::splat(false)
    }

    #[inline]
    pub fn all(self) -> bool {
        self == Self::splat(true)
    }
}

impl<Element, const LANES: usize> core::ops::BitAnd for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    <LaneCount<LANES> as SupportedLaneCount>::BitMask: AsRef<[u8]> + AsMut<[u8]>,
{
    type Output = Self;
    #[inline]
    fn bitand(mut self, rhs: Self) -> Self {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l &= r;
        }
        self
    }
}

impl<Element, const LANES: usize> core::ops::BitOr for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    <LaneCount<LANES> as SupportedLaneCount>::BitMask: AsRef<[u8]> + AsMut<[u8]>,
{
    type Output = Self;
    #[inline]
    fn bitor(mut self, rhs: Self) -> Self {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l |= r;
        }
        self
    }
}

impl<Element, const LANES: usize> core::ops::BitXor for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l ^= r;
        }
        self
    }
}

impl<Element, const LANES: usize> core::ops::Not for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn not(mut self) -> Self::Output {
        for x in self.0.as_mut() {
            *x = !*x;
        }
        if LANES % 8 > 0 {
            *self.0.as_mut().last_mut().unwrap() &= u8::MAX >> (8 - LANES % 8);
        }
        self
    }
}
