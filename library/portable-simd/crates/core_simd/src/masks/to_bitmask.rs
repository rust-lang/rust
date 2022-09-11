use super::{mask_impl, Mask, MaskElement};
use crate::simd::{LaneCount, SupportedLaneCount};

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
#[cfg(feature = "generic_const_exprs")]
pub trait ToBitMaskArray: Sealed {
    /// The length of the bitmask array.
    const BYTES: usize;

    /// Converts a mask to a bitmask.
    fn to_bitmask_array(self) -> [u8; Self::BYTES];

    /// Converts a bitmask to a mask.
    fn from_bitmask_array(bitmask: [u8; Self::BYTES]) -> Self;
}

macro_rules! impl_integer_intrinsic {
    { $(impl ToBitMask<BitMask=$int:ty> for Mask<_, $lanes:literal>)* } => {
        $(
        impl<T: MaskElement> ToBitMask for Mask<T, $lanes> {
            type BitMask = $int;

            fn to_bitmask(self) -> $int {
                self.0.to_bitmask_integer()
            }

            fn from_bitmask(bitmask: $int) -> Self {
                Self(mask_impl::Mask::from_bitmask_integer(bitmask))
            }
        }
        )*
    }
}

impl_integer_intrinsic! {
    impl ToBitMask<BitMask=u8> for Mask<_, 1>
    impl ToBitMask<BitMask=u8> for Mask<_, 2>
    impl ToBitMask<BitMask=u8> for Mask<_, 4>
    impl ToBitMask<BitMask=u8> for Mask<_, 8>
    impl ToBitMask<BitMask=u16> for Mask<_, 16>
    impl ToBitMask<BitMask=u32> for Mask<_, 32>
    impl ToBitMask<BitMask=u64> for Mask<_, 64>
}

/// Returns the minimum number of bytes in a bitmask with `lanes` lanes.
#[cfg(feature = "generic_const_exprs")]
pub const fn bitmask_len(lanes: usize) -> usize {
    (lanes + 7) / 8
}

#[cfg(feature = "generic_const_exprs")]
impl<T: MaskElement, const LANES: usize> ToBitMaskArray for Mask<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const BYTES: usize = bitmask_len(LANES);

    fn to_bitmask_array(self) -> [u8; Self::BYTES] {
        self.0.to_bitmask_array()
    }

    fn from_bitmask_array(bitmask: [u8; Self::BYTES]) -> Self {
        Mask(mask_impl::Mask::from_bitmask_array(bitmask))
    }
}
