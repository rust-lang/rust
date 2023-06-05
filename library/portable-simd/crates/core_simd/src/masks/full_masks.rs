//! Masks that take up full SIMD vector registers.

use super::MaskElement;
use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SupportedLaneCount, ToBitMask};

#[cfg(feature = "generic_const_exprs")]
use crate::simd::ToBitMaskArray;

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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const LANES: usize> PartialEq for Mask<T, LANES>
where
    T: MaskElement + PartialEq,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const LANES: usize> PartialOrd for Mask<T, LANES>
where
    T: MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
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

impl<T, const LANES: usize> Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
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
    pub fn to_int(self) -> Simd<T, LANES> {
        self.0
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub unsafe fn from_int_unchecked(value: Simd<T, LANES>) -> Self {
        Self(value)
    }

    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn convert<U>(self) -> Mask<U, LANES>
    where
        U: MaskElement,
    {
        // Safety: masks are simply integer vectors of 0 and -1, and we can cast the element type.
        unsafe { Mask(intrinsics::simd_cast(self.0)) }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn to_bitmask_array<const N: usize>(self) -> [u8; N]
    where
        super::Mask<T, LANES>: ToBitMaskArray,
        [(); <super::Mask<T, LANES> as ToBitMaskArray>::BYTES]: Sized,
    {
        assert_eq!(<super::Mask<T, LANES> as ToBitMaskArray>::BYTES, N);

        // Safety: N is the correct bitmask size
        unsafe {
            // Compute the bitmask
            let bitmask: [u8; <super::Mask<T, LANES> as ToBitMaskArray>::BYTES] =
                intrinsics::simd_bitmask(self.0);

            // Transmute to the return type, previously asserted to be the same size
            let mut bitmask: [u8; N] = core::mem::transmute_copy(&bitmask);

            // LLVM assumes bit order should match endianness
            if cfg!(target_endian = "big") {
                for x in bitmask.as_mut() {
                    *x = x.reverse_bits();
                }
            };

            bitmask
        }
    }

    #[cfg(feature = "generic_const_exprs")]
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn from_bitmask_array<const N: usize>(mut bitmask: [u8; N]) -> Self
    where
        super::Mask<T, LANES>: ToBitMaskArray,
        [(); <super::Mask<T, LANES> as ToBitMaskArray>::BYTES]: Sized,
    {
        assert_eq!(<super::Mask<T, LANES> as ToBitMaskArray>::BYTES, N);

        // Safety: N is the correct bitmask size
        unsafe {
            // LLVM assumes bit order should match endianness
            if cfg!(target_endian = "big") {
                for x in bitmask.as_mut() {
                    *x = x.reverse_bits();
                }
            }

            // Transmute to the bitmask type, previously asserted to be the same size
            let bitmask: [u8; <super::Mask<T, LANES> as ToBitMaskArray>::BYTES] =
                core::mem::transmute_copy(&bitmask);

            // Compute the regular mask
            Self::from_int_unchecked(intrinsics::simd_select_bitmask(
                bitmask,
                Self::splat(true).to_int(),
                Self::splat(false).to_int(),
            ))
        }
    }

    #[inline]
    pub(crate) fn to_bitmask_integer<U: ReverseBits>(self) -> U
    where
        super::Mask<T, LANES>: ToBitMask<BitMask = U>,
    {
        // Safety: U is required to be the appropriate bitmask type
        let bitmask: U = unsafe { intrinsics::simd_bitmask(self.0) };

        // LLVM assumes bit order should match endianness
        if cfg!(target_endian = "big") {
            bitmask.reverse_bits(LANES)
        } else {
            bitmask
        }
    }

    #[inline]
    pub(crate) fn from_bitmask_integer<U: ReverseBits>(bitmask: U) -> Self
    where
        super::Mask<T, LANES>: ToBitMask<BitMask = U>,
    {
        // LLVM assumes bit order should match endianness
        let bitmask = if cfg!(target_endian = "big") {
            bitmask.reverse_bits(LANES)
        } else {
            bitmask
        };

        // Safety: U is required to be the appropriate bitmask type
        unsafe {
            Self::from_int_unchecked(intrinsics::simd_select_bitmask(
                bitmask,
                Self::splat(true).to_int(),
                Self::splat(false).to_int(),
            ))
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

impl<T, const LANES: usize> From<Mask<T, LANES>> for Simd<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitxor(self, rhs: Self) -> Self {
        // Safety: `self` is an integer vector
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
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn not(self) -> Self::Output {
        Self::splat(true) ^ self
    }
}
