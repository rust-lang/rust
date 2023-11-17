//! Masks that take up full SIMD vector registers.

use crate::simd::intrinsics;
use crate::simd::{LaneCount, MaskElement, Simd, SupportedLaneCount};

#[repr(transparent)]
pub struct Mask<T, const N: usize>(Simd<T, N>)
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
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
                let bitsize = core::mem::size_of::<$int>() * 8;
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
    pub fn splat(value: bool) -> Self {
        Self(Simd::splat(if value { T::TRUE } else { T::FALSE }))
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        T::eq(self.0[lane], T::TRUE)
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0[lane] = if value { T::TRUE } else { T::FALSE }
    }

    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn to_int(self) -> Simd<T, N> {
        self.0
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub unsafe fn from_int_unchecked(value: Simd<T, N>) -> Self {
        Self(value)
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn convert<U>(self) -> Mask<U, N>
    where
        U: MaskElement,
    {
        // Safety: masks are simply integer vectors of 0 and -1, and we can cast the element type.
        unsafe { Mask(intrinsics::simd_cast(self.0)) }
    }

    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn to_bitmask_vector(self) -> Simd<T, N> {
        let mut bitmask = Self::splat(false).to_int();

        // Safety: Bytes is the right size array
        unsafe {
            // Compute the bitmask
            let mut bytes: <LaneCount<N> as SupportedLaneCount>::BitMask =
                intrinsics::simd_bitmask(self.0);

            // LLVM assumes bit order should match endianness
            if cfg!(target_endian = "big") {
                for x in bytes.as_mut() {
                    *x = x.reverse_bits()
                }
            }

            assert!(
                core::mem::size_of::<Simd<T, N>>()
                    >= core::mem::size_of::<<LaneCount<N> as SupportedLaneCount>::BitMask>()
            );
            core::ptr::copy_nonoverlapping(
                bytes.as_ref().as_ptr(),
                bitmask.as_mut_array().as_mut_ptr() as _,
                bytes.as_ref().len(),
            );
        }

        bitmask
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn from_bitmask_vector(bitmask: Simd<T, N>) -> Self {
        let mut bytes = <LaneCount<N> as SupportedLaneCount>::BitMask::default();

        // Safety: Bytes is the right size array
        unsafe {
            assert!(
                core::mem::size_of::<Simd<T, N>>()
                    >= core::mem::size_of::<<LaneCount<N> as SupportedLaneCount>::BitMask>()
            );
            core::ptr::copy_nonoverlapping(
                bitmask.as_array().as_ptr() as _,
                bytes.as_mut().as_mut_ptr(),
                bytes.as_mut().len(),
            );

            // LLVM assumes bit order should match endianness
            if cfg!(target_endian = "big") {
                for x in bytes.as_mut() {
                    *x = x.reverse_bits();
                }
            }

            // Compute the regular mask
            Self::from_int_unchecked(intrinsics::simd_select_bitmask(
                bytes,
                Self::splat(true).to_int(),
                Self::splat(false).to_int(),
            ))
        }
    }

    #[inline]
    unsafe fn to_bitmask_impl<U: ReverseBits, const M: usize>(self) -> U
    where
        LaneCount<M>: SupportedLaneCount,
    {
        let resized = self.to_int().resize::<M>(T::FALSE);

        // Safety: `resized` is an integer vector with length M, which must match T
        let bitmask: U = unsafe { intrinsics::simd_bitmask(resized) };

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
            intrinsics::simd_select_bitmask(
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
        macro_rules! bitmask {
            { $($ty:ty: $($len:literal),*;)* } => {
                match N {
                    $($(
                    // Safety: bitmask matches length
                    $len => unsafe { self.to_bitmask_impl::<$ty, $len>() as u64 },
                    )*)*
                    // Safety: bitmask matches length
                    _ => unsafe { self.to_bitmask_impl::<u64, 64>() },
                }
            }
        }
        #[cfg(all_lane_counts)]
        bitmask! {
            u8: 1, 2, 3, 4, 5, 6, 7, 8;
            u16: 9, 10, 11, 12, 13, 14, 15, 16;
            u32: 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32;
            u64: 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64;
        }
        #[cfg(not(all_lane_counts))]
        bitmask! {
            u8: 1, 2, 4, 8;
            u16: 16;
            u32: 32;
            u64: 64;
        }
    }

    #[inline]
    pub(crate) fn from_bitmask_integer(bitmask: u64) -> Self {
        // TODO modify simd_bitmask_select to truncate input, making this unnecessary
        macro_rules! bitmask {
            { $($ty:ty: $($len:literal),*;)* } => {
                match N {
                    $($(
                    // Safety: bitmask matches length
                    $len => unsafe { Self::from_bitmask_impl::<$ty, $len>(bitmask as $ty) },
                    )*)*
                    // Safety: bitmask matches length
                    _ => unsafe { Self::from_bitmask_impl::<u64, 64>(bitmask) },
                }
            }
        }
        #[cfg(all_lane_counts)]
        bitmask! {
            u8: 1, 2, 3, 4, 5, 6, 7, 8;
            u16: 9, 10, 11, 12, 13, 14, 15, 16;
            u32: 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32;
            u64: 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64;
        }
        #[cfg(not(all_lane_counts))]
        bitmask! {
            u8: 1, 2, 4, 8;
            u16: 16;
            u32: 32;
            u64: 64;
        }
    }

    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub fn any(self) -> bool {
        // Safety: use `self` as an integer vector
        unsafe { intrinsics::simd_reduce_any(self.to_int()) }
    }

    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn all(self) -> bool {
        // Safety: use `self` as an integer vector
        unsafe { intrinsics::simd_reduce_all(self.to_int()) }
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(intrinsics::simd_and(self.0, rhs.0)) }
    }
}

impl<T, const N: usize> core::ops::BitOr for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(intrinsics::simd_or(self.0, rhs.0)) }
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
    fn bitxor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
        unsafe { Self(intrinsics::simd_xor(self.0, rhs.0)) }
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
    fn not(self) -> Self::Output {
        Self::splat(true) ^ self
    }
}
