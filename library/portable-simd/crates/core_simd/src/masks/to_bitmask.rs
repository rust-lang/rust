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
///
/// # Safety
/// This trait is `unsafe` and sealed, since the `BitMask` type must match the number of lanes in
/// the mask.
pub unsafe trait ToBitMask: Sealed {
    /// The integer bitmask type.
    type BitMask;

    /// Converts a mask to a bitmask.
    fn to_bitmask(self) -> Self::BitMask;

    /// Converts a bitmask to a mask.
    fn from_bitmask(bitmask: Self::BitMask) -> Self;
}

macro_rules! impl_integer_intrinsic {
    { $(unsafe impl ToBitMask<BitMask=$int:ty> for Mask<_, $lanes:literal>)* } => {
        $(
        unsafe impl<T: MaskElement> ToBitMask for Mask<T, $lanes> {
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
    unsafe impl ToBitMask<BitMask=u8> for Mask<_, 8>
    unsafe impl ToBitMask<BitMask=u16> for Mask<_, 16>
    unsafe impl ToBitMask<BitMask=u32> for Mask<_, 32>
    unsafe impl ToBitMask<BitMask=u64> for Mask<_, 64>
}
