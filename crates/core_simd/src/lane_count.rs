mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Specifies the number of lanes in a SIMD vector as a type.
pub struct LaneCount<const N: usize>;

impl<const N: usize> LaneCount<N> {
    /// The number of bytes in a bitmask with this many lanes.
    pub const BITMASK_LEN: usize = (N + 7) / 8;
}

/// Statically guarantees that a lane count is marked as supported.
///
/// This trait is *sealed*: the list of implementors below is total.
/// Users do not have the ability to mark additional `LaneCount<N>` values as supported.
/// Only SIMD vectors with supported lane counts are constructable.
pub trait SupportedLaneCount: Sealed {
    #[doc(hidden)]
    type BitMask: Copy + AsRef<[u8]> + AsMut<[u8]>;
    #[doc(hidden)]
    const EMPTY_BIT_MASK: Self::BitMask;
    #[doc(hidden)]
    const FULL_BIT_MASK: Self::BitMask;
}

impl<const N: usize> Sealed for LaneCount<N> {}

macro_rules! supported_lane_count {
    ($($lanes:literal),+) => {
        $(
            impl SupportedLaneCount for LaneCount<$lanes> {
                type BitMask = [u8; ($lanes + 7) / 8];
                const EMPTY_BIT_MASK: Self::BitMask = [0; ($lanes + 7) / 8];
                const FULL_BIT_MASK: Self::BitMask = {
                    const LEN: usize = ($lanes + 7) / 8;
                    let mut array = [!0u8; LEN];
                    if $lanes % 8 > 0 {
                        array[LEN - 1] = (!0) >> (8 - $lanes % 8);
                    }
                    array
                };
            }
        )+
    };
}

supported_lane_count!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
);
