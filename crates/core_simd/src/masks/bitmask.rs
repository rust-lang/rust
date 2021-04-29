use crate::Mask;
use core::marker::PhantomData;

/// Helper trait for limiting int conversion types
pub trait ConvertToInt {}
impl<const LANES: usize> ConvertToInt for crate::SimdI8<LANES> where Self: crate::LanesAtMost32 {}
impl<const LANES: usize> ConvertToInt for crate::SimdI16<LANES> where Self: crate::LanesAtMost32 {}
impl<const LANES: usize> ConvertToInt for crate::SimdI32<LANES> where Self: crate::LanesAtMost32 {}
impl<const LANES: usize> ConvertToInt for crate::SimdI64<LANES> where Self: crate::LanesAtMost32 {}
impl<const LANES: usize> ConvertToInt for crate::SimdIsize<LANES> where Self: crate::LanesAtMost32 {}

/// A mask where each lane is represented by a single bit.
#[repr(transparent)]
pub struct BitMask<T: Mask, const LANES: usize>(T::BitMask, PhantomData<[(); LANES]>);

impl<T: Mask, const LANES: usize> Copy for BitMask<T, LANES> {}

impl<T: Mask, const LANES: usize> Clone for BitMask<T, LANES> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Mask, const LANES: usize> PartialEq for BitMask<T, LANES> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<T: Mask, const LANES: usize> PartialOrd for BitMask<T, LANES> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.as_ref().partial_cmp(other.0.as_ref())
    }
}

impl<T: Mask, const LANES: usize> Eq for BitMask<T, LANES> {}

impl<T: Mask, const LANES: usize> Ord for BitMask<T, LANES> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl<T: Mask, const LANES: usize> BitMask<T, LANES> {
    #[inline]
    pub fn splat(value: bool) -> Self {
        let mut mask = T::BitMask::default();
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
        (self.0.as_ref()[lane / 8] >> lane % 8) & 0x1 > 0
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
            let mask: T::IntBitMask = core::mem::transmute_copy(&self);
            crate::intrinsics::simd_select_bitmask(mask, !V::default(), V::default())
        }
    }

    #[inline]
    pub unsafe fn from_int_unchecked<V>(value: V) -> Self
    where
        V: crate::LanesAtMost32,
    {
        // TODO remove the transmute when rustc is more flexible
        assert_eq!(
            core::mem::size_of::<T::IntBitMask>(),
            core::mem::size_of::<T::BitMask>()
        );
        let mask: T::IntBitMask = crate::intrinsics::simd_bitmask(value);
        Self(core::mem::transmute_copy(&mask), PhantomData)
    }

    #[inline]
    pub fn to_bitmask<U: Mask>(self) -> U::BitMask {
        assert_eq!(
            core::mem::size_of::<T::BitMask>(),
            core::mem::size_of::<U::BitMask>()
        );
        unsafe { core::mem::transmute_copy(&self.0) }
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

impl<T: Mask, const LANES: usize> core::ops::BitAnd for BitMask<T, LANES>
where
    T::BitMask: Default + AsRef<[u8]> + AsMut<[u8]>,
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

impl<T: Mask, const LANES: usize> core::ops::BitOr for BitMask<T, LANES>
where
    T::BitMask: Default + AsRef<[u8]> + AsMut<[u8]>,
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

impl<T: Mask, const LANES: usize> core::ops::BitXor for BitMask<T, LANES> {
    type Output = Self;
    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        for (l, r) in self.0.as_mut().iter_mut().zip(rhs.0.as_ref().iter()) {
            *l ^= r;
        }
        self
    }
}

impl<T: Mask, const LANES: usize> core::ops::Not for BitMask<T, LANES> {
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

pub type Mask8<T, const LANES: usize> = BitMask<T, LANES>;
pub type Mask16<T, const LANES: usize> = BitMask<T, LANES>;
pub type Mask32<T, const LANES: usize> = BitMask<T, LANES>;
pub type Mask64<T, const LANES: usize> = BitMask<T, LANES>;
pub type MaskSize<T, const LANES: usize> = BitMask<T, LANES>;
