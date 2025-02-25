#![allow(unused_imports)]
use super::MaskElement;
use crate::simd::{LaneCount, Simd, SupportedLaneCount};
use core::marker::PhantomData;

/// A mask where each lane is represented by a single bit.
#[repr(transparent)]
pub struct Mask<T, const N: usize>(
    <LaneCount<N> as SupportedLaneCount>::BitMask,
    PhantomData<T>,
)
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount;

impl<T, const N: usize> Copy for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
}

impl<T, const N: usize> Clone for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> PartialEq for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<T, const N: usize> PartialOrd for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.as_ref().partial_cmp(other.0.as_ref())
    }
}

impl<T, const N: usize> Eq for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
}

impl<T, const N: usize> Ord for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl<T, const N: usize> Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn splat(value: bool) -> Self {
        let mut mask = <LaneCount<N> as SupportedLaneCount>::BitMask::default();
        if value {
            mask.as_mut().fill(u8::MAX)
        } else {
            mask.as_mut().fill(u8::MIN)
        }
        if N % 8 > 0 {
            *mask.as_mut().last_mut().unwrap() &= u8::MAX >> (8 - N % 8);
        }
        Self(mask, PhantomData)
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
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
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn to_int(self) -> Simd<T, N> {
        unsafe {
            core::intrinsics::simd::simd_select_bitmask(
                self.0,
                Simd::splat(T::TRUE),
                Simd::splat(T::FALSE),
            )
        }
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub unsafe fn from_int_unchecked(value: Simd<T, N>) -> Self {
        unsafe { Self(core::intrinsics::simd::simd_bitmask(value), PhantomData) }
    }

    #[inline]
    pub fn to_bitmask_integer(self) -> u64 {
        let mut bitmask = [0u8; 8];
        bitmask[..self.0.as_ref().len()].copy_from_slice(self.0.as_ref());
        u64::from_ne_bytes(bitmask)
    }

    #[inline]
    pub fn from_bitmask_integer(bitmask: u64) -> Self {
        let mut bytes = <LaneCount<N> as SupportedLaneCount>::BitMask::default();
        let len = bytes.as_mut().len();
        bytes
            .as_mut()
            .copy_from_slice(&bitmask.to_ne_bytes()[..len]);
        Self(bytes, PhantomData)
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn convert<U>(self) -> Mask<U, N>
    where
        U: MaskElement,
    {
        // Safety: bitmask layout does not depend on the element width
        unsafe { core::mem::transmute_copy(&self) }
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub fn any(self) -> bool {
        self != Self::splat(false)
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub fn all(self) -> bool {
        self == Self::splat(true)
    }
}

impl<T, const N: usize> core::ops::BitAnd for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
    <LaneCount<N> as SupportedLaneCount>::BitMask: AsRef<[u8]> + AsMut<[u8]>,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(mut self, rhs: Self) -> Self {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l &= r;
        }
        self
    }
}

impl<T, const N: usize> core::ops::BitOr for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
    <LaneCount<N> as SupportedLaneCount>::BitMask: AsRef<[u8]> + AsMut<[u8]>,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(mut self, rhs: Self) -> Self {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l |= r;
        }
        self
    }
}

impl<T, const N: usize> core::ops::BitXor for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l ^= r;
        }
        self
    }
}

impl<T, const N: usize> core::ops::Not for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn not(mut self) -> Self::Output {
        for x in self.0.as_mut() {
            *x = !*x;
        }
        if N % 8 > 0 {
            *self.0.as_mut().last_mut().unwrap() &= u8::MAX >> (8 - N % 8);
        }
        self
    }
}
