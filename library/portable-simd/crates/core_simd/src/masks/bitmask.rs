use super::MaskElement;
use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SupportedLaneCount};
use core::marker::PhantomData;

/// A mask where each lane is represented by a single bit.
#[repr(transparent)]
pub struct Mask<T, const LANES: usize>(
    <LaneCount<LANES> as SupportedLaneCount>::BitMask,
    PhantomData<T>,
)
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
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const LANES: usize> PartialEq for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<T, const LANES: usize> PartialOrd for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.as_ref().partial_cmp(other.0.as_ref())
    }
}

impl<T, const LANES: usize> Eq for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Ord for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl<T, const LANES: usize> Mask<T, LANES>
where
    T: MaskElement,
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
        unsafe {
            self.0.as_mut()[lane / 8] ^= ((value ^ self.test_unchecked(lane)) as u8) << (lane % 8)
        }
    }

    #[inline]
    pub fn to_int(self) -> Simd<T, LANES> {
        unsafe {
            let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                core::mem::transmute_copy(&self);
            intrinsics::simd_select_bitmask(mask, Simd::splat(T::TRUE), Simd::splat(T::FALSE))
        }
    }

    #[inline]
    pub unsafe fn from_int_unchecked(value: Simd<T, LANES>) -> Self {
        // TODO remove the transmute when rustc is more flexible
        assert_eq!(
            core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::BitMask>(),
            core::mem::size_of::<<LaneCount::<LANES> as SupportedLaneCount>::IntBitMask>(),
        );
        unsafe {
            let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                intrinsics::simd_bitmask(value);
            Self(core::mem::transmute_copy(&mask), PhantomData)
        }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        // Safety: these are the same type and we are laundering the generic
        unsafe { core::mem::transmute_copy(&self.0) }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    pub fn from_bitmask(bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        // Safety: these are the same type and we are laundering the generic
        Self(unsafe { core::mem::transmute_copy(&bitmask) }, PhantomData)
    }

    #[inline]
    pub fn convert<U>(self) -> Mask<U, LANES>
    where
        U: MaskElement,
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

impl<T, const LANES: usize> core::ops::BitAnd for Mask<T, LANES>
where
    T: MaskElement,
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

impl<T, const LANES: usize> core::ops::BitOr for Mask<T, LANES>
where
    T: MaskElement,
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

impl<T, const LANES: usize> core::ops::BitXor for Mask<T, LANES>
where
    T: MaskElement,
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

impl<T, const LANES: usize> core::ops::Not for Mask<T, LANES>
where
    T: MaskElement,
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
