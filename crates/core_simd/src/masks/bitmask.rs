use crate::{LaneCount, SupportedLaneCount};

/// Helper trait for limiting int conversion types
pub trait ConvertToInt {}
impl<const LANES: usize> ConvertToInt for crate::SimdI8<LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}
impl<const LANES: usize> ConvertToInt for crate::SimdI16<LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}
impl<const LANES: usize> ConvertToInt for crate::SimdI32<LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}
impl<const LANES: usize> ConvertToInt for crate::SimdI64<LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}
impl<const LANES: usize> ConvertToInt for crate::SimdIsize<LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}

/// A mask where each lane is represented by a single bit.
#[repr(transparent)]
pub struct BitMask<const LANES: usize>(<LaneCount<LANES> as SupportedLaneCount>::BitMask)
where
    LaneCount<LANES>: SupportedLaneCount;

impl<const LANES: usize> Copy for BitMask<LANES> where LaneCount<LANES>: SupportedLaneCount {}

impl<const LANES: usize> Clone for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<const LANES: usize> PartialEq for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<const LANES: usize> PartialOrd for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.as_ref().partial_cmp(other.0.as_ref())
    }
}

impl<const LANES: usize> Eq for BitMask<LANES> where LaneCount<LANES>: SupportedLaneCount {}

impl<const LANES: usize> Ord for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl<const LANES: usize> BitMask<LANES>
where
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
        Self(mask)
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
    pub fn to_int<V>(self) -> V
    where
        V: ConvertToInt + Default + core::ops::Not<Output = V>,
    {
        unsafe {
            let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
                core::mem::transmute_copy(&self);
            crate::intrinsics::simd_select_bitmask(mask, !V::default(), V::default())
        }
    }

    #[inline]
    pub unsafe fn from_int_unchecked<V>(value: V) -> Self
    where
        V: crate::Vector,
    {
        // TODO remove the transmute when rustc is more flexible
        assert_eq!(
            core::mem::size_of::<<crate::LaneCount::<LANES> as crate::SupportedLaneCount>::BitMask>(
            ),
            core::mem::size_of::<
                <crate::LaneCount::<LANES> as crate::SupportedLaneCount>::IntBitMask,
            >(),
        );
        let mask: <LaneCount<LANES> as SupportedLaneCount>::IntBitMask =
            crate::intrinsics::simd_bitmask(value);
        Self(core::mem::transmute_copy(&mask))
    }

    #[inline]
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        // Safety: these are the same type and we are laundering the generic
        unsafe { core::mem::transmute_copy(&self.0) }
    }

    #[inline]
    pub fn from_bitmask(bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        // Safety: these are the same type and we are laundering the generic
        Self(unsafe { core::mem::transmute_copy(&bitmask) })
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

impl<const LANES: usize> core::ops::BitAnd for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    <LaneCount<LANES> as SupportedLaneCount>::BitMask: Default + AsRef<[u8]> + AsMut<[u8]>,
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

impl<const LANES: usize> core::ops::BitOr for BitMask<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    <LaneCount<LANES> as SupportedLaneCount>::BitMask: Default + AsRef<[u8]> + AsMut<[u8]>,
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

impl<const LANES: usize> core::ops::BitXor for BitMask<LANES>
where
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

impl<const LANES: usize> core::ops::Not for BitMask<LANES>
where
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

pub type Mask8<const LANES: usize> = BitMask<LANES>;
pub type Mask16<const LANES: usize> = BitMask<LANES>;
pub type Mask32<const LANES: usize> = BitMask<LANES>;
pub type Mask64<const LANES: usize> = BitMask<LANES>;
pub type MaskSize<const LANES: usize> = BitMask<LANES>;
