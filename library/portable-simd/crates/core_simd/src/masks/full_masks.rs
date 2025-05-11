//! Masks that take up full SIMD vector registers.

use crate::simd::{LaneCount, MaskElement, Simd, SupportedLaneCount};

#[repr(transparent)]
pub(crate) struct Mask<T, const N: usize>(Simd<T, N>)
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
    T: MaskElement + PartialEq,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const N: usize> PartialOrd for Mask<T, N>
where
    T: MaskElement + PartialOrd,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const N: usize> Eq for Mask<T, N>
where
    T: MaskElement + Eq,
    LaneCount<N>: SupportedLaneCount,
{
}

impl<T, const N: usize> Ord for Mask<T, N>
where
    T: MaskElement + Ord,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// Used for bitmask bit order workaround
pub(crate) trait ReverseBits {
    // Reverse the least significant `n` bits of `self`.
    // (Remaining bits must be 0.)
    fn reverse_bits(self, n: usize) -> Self;
}

macro_rules! impl_reverse_bits {
    { $($int:ty),* } => {
        $(
        impl ReverseBits for $int {
            #[inline(always)]
            fn reverse_bits(self, n: usize) -> Self {
                let rev = <$int>::reverse_bits(self);
                let bitsize = size_of::<$int>() * 8;
                if n < bitsize {
                    // Shift things back to the right
                    rev >> (bitsize - n)
                } else {
                    rev
                }
            }
        }
        )*
    }
}

impl_reverse_bits! { u8, u16, u32, u64 }

impl<T, const N: usize> Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub(crate) fn splat(value: bool) -> Self {
        Self(Simd::splat(if value { T::TRUE } else { T::FALSE }))
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub(crate) unsafe fn test_unchecked(&self, lane: usize) -> bool {
        T::eq(self.0[lane], T::TRUE)
    }

    #[inline]
    pub(crate) unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0[lane] = if value { T::TRUE } else { T::FALSE }
    }

    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub(crate) fn to_int(self) -> Simd<T, N> {
        self.0
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub(crate) unsafe fn from_int_unchecked(value: Simd<T, N>) -> Self {
        Self(value)
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub(crate) fn convert<U>(self) -> Mask<U, N>
    where
        U: MaskElement,
    {
        // Safety: masks are simply integer vectors of 0 and -1, and we can cast the element type.
        unsafe { Mask(core::intrinsics::simd::simd_cast(self.0)) }
    }

    #[inline]
    unsafe fn to_bitmask_impl<U: ReverseBits, const M: usize>(self) -> U
    where
        LaneCount<M>: SupportedLaneCount,
    {
        let resized = self.to_int().resize::<M>(T::FALSE);

        // Safety: `resized` is an integer vector with length M, which must match T
        let bitmask: U = unsafe { core::intrinsics::simd::simd_bitmask(resized) };

        // LLVM assumes bit order should match endianness
        if cfg!(target_endian = "big") {
            bitmask.reverse_bits(M)
        } else {
            bitmask
        }
    }

    #[inline]
    unsafe fn from_bitmask_impl<U: ReverseBits, const M: usize>(bitmask: U) -> Self
    where
        LaneCount<M>: SupportedLaneCount,
    {
        // LLVM assumes bit order should match endianness
        let bitmask = if cfg!(target_endian = "big") {
            bitmask.reverse_bits(M)
        } else {
            bitmask
        };

        // SAFETY: `mask` is the correct bitmask type for a u64 bitmask
        let mask: Simd<T, M> = unsafe {
            core::intrinsics::simd::simd_select_bitmask(
                bitmask,
                Simd::<T, M>::splat(T::TRUE),
                Simd::<T, M>::splat(T::FALSE),
            )
        };

        // SAFETY: `mask` only contains `T::TRUE` or `T::FALSE`
        unsafe { Self::from_int_unchecked(mask.resize::<N>(T::FALSE)) }
    }

    #[inline]
    pub(crate) fn to_bitmask_integer(self) -> u64 {
        // TODO modify simd_bitmask to zero-extend output, making this unnecessary
        if N <= 8 {
            // Safety: bitmask matches length
            unsafe { self.to_bitmask_impl::<u8, 8>() as u64 }
        } else if N <= 16 {
            // Safety: bitmask matches length
            unsafe { self.to_bitmask_impl::<u16, 16>() as u64 }
        } else if N <= 32 {
            // Safety: bitmask matches length
            unsafe { self.to_bitmask_impl::<u32, 32>() as u64 }
        } else {
            // Safety: bitmask matches length
            unsafe { self.to_bitmask_impl::<u64, 64>() }
        }
    }

    #[inline]
    pub(crate) fn from_bitmask_integer(bitmask: u64) -> Self {
        // TODO modify simd_bitmask_select to truncate input, making this unnecessary
        if N <= 8 {
            // Safety: bitmask matches length
            unsafe { Self::from_bitmask_impl::<u8, 8>(bitmask as u8) }
        } else if N <= 16 {
            // Safety: bitmask matches length
            unsafe { Self::from_bitmask_impl::<u16, 16>(bitmask as u16) }
        } else if N <= 32 {
            // Safety: bitmask matches length
            unsafe { Self::from_bitmask_impl::<u32, 32>(bitmask as u32) }
        } else {
            // Safety: bitmask matches length
            unsafe { Self::from_bitmask_impl::<u64, 64>(bitmask) }
        }
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub(crate) fn any(self) -> bool {
        // Safety: use `self` as an integer vector
        unsafe { core::intrinsics::simd::simd_reduce_any(self.to_int()) }
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub(crate) fn all(self) -> bool {
        // Safety: use `self` as an integer vector
        unsafe { core::intrinsics::simd::simd_reduce_all(self.to_int()) }
    }
}

impl<T, const N: usize> From<Mask<T, N>> for Simd<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn from(value: Mask<T, N>) -> Self {
        value.0
    }
}

impl<T, const N: usize> core::ops::BitAnd for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(core::intrinsics::simd::simd_and(self.0, rhs.0)) }
    }
}

impl<T, const N: usize> core::ops::BitOr for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(core::intrinsics::simd::simd_or(self.0, rhs.0)) }
    }
}

impl<T, const N: usize> core::ops::BitXor for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(core::intrinsics::simd::simd_xor(self.0, rhs.0)) }
    }
}

impl<T, const N: usize> core::ops::Not for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        Self::splat(true) ^ self
    }
}
