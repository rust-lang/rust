use super::{mask_impl, Mask, MaskElement};
use crate::simd::{LaneCount, SupportedLaneCount};
use core::borrow::{Borrow, BorrowMut};

mod sealed {
    pub trait Sealed {}
}
pub use sealed::Sealed;

impl<T, const LANES: usize> Sealed for Mask<T, LANES>
where
    T: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

/// Converts masks to and from integer bitmasks.
///
/// Each bit of the bitmask corresponds to a mask lane, starting with the LSB.
pub trait ToBitMask: Sealed {
    /// The integer bitmask type.
    type BitMask;

    /// Converts a mask to a bitmask.
    fn to_bitmask(self) -> Self::BitMask;

    /// Converts a bitmask to a mask.
    fn from_bitmask(bitmask: Self::BitMask) -> Self;
}

/// Converts masks to and from byte array bitmasks.
///
/// Each bit of the bitmask corresponds to a mask lane, starting with the LSB of the first byte.
pub trait ToBitMaskArray: Sealed {
    /// The bitmask array.
    type BitMaskArray: Copy
        + Unpin
        + Send
        + Sync
        + AsRef<[u8]>
        + AsMut<[u8]>
        + Borrow<[u8]>
        + BorrowMut<[u8]>
        + 'static;

    /// Converts a mask to a bitmask.
    fn to_bitmask_array(self) -> Self::BitMaskArray;

    /// Converts a bitmask to a mask.
    fn from_bitmask_array(bitmask: Self::BitMaskArray) -> Self;
}

macro_rules! impl_integer {
    { $(impl ToBitMask<BitMask=$int:ty> for Mask<_, $lanes:literal>)* } => {
        $(
        impl<T: MaskElement> ToBitMask for Mask<T, $lanes> {
            type BitMask = $int;

            #[inline]
            fn to_bitmask(self) -> $int {
                self.0.to_bitmask_integer()
            }

            #[inline]
            fn from_bitmask(bitmask: $int) -> Self {
                Self(mask_impl::Mask::from_bitmask_integer(bitmask))
            }
        }
        )*
    }
}

macro_rules! impl_array {
    { $(impl ToBitMaskArray<Bytes=$int:literal> for Mask<_, $lanes:literal>)* } => {
        $(
        impl<T: MaskElement> ToBitMaskArray for Mask<T, $lanes> {
            type BitMaskArray = [u8; $int];

            #[inline]
            fn to_bitmask_array(self) -> Self::BitMaskArray {
                self.0.to_bitmask_array()
            }

            #[inline]
            fn from_bitmask_array(bitmask: Self::BitMaskArray) -> Self {
                Self(mask_impl::Mask::from_bitmask_array(bitmask))
            }
        }
        )*
    }
}

impl_integer! {
    impl ToBitMask<BitMask=u8> for Mask<_, 1>
    impl ToBitMask<BitMask=u8> for Mask<_, 2>
    impl ToBitMask<BitMask=u8> for Mask<_, 4>
    impl ToBitMask<BitMask=u8> for Mask<_, 8>
    impl ToBitMask<BitMask=u16> for Mask<_, 16>
    impl ToBitMask<BitMask=u32> for Mask<_, 32>
    impl ToBitMask<BitMask=u64> for Mask<_, 64>
}

impl_array! {
    impl ToBitMaskArray<Bytes=1> for Mask<_, 1>
    impl ToBitMaskArray<Bytes=1> for Mask<_, 2>
    impl ToBitMaskArray<Bytes=1> for Mask<_, 4>
    impl ToBitMaskArray<Bytes=1> for Mask<_, 8>
    impl ToBitMaskArray<Bytes=2> for Mask<_, 16>
    impl ToBitMaskArray<Bytes=4> for Mask<_, 32>
    impl ToBitMaskArray<Bytes=8> for Mask<_, 64>
}
