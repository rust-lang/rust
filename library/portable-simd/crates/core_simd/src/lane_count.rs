mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Specifies the number of lanes in a SIMD vector as a type.
pub struct LaneCount<const LANES: usize>;

impl<const LANES: usize> LaneCount<LANES> {
    /// The number of bytes in a bitmask with this many lanes.
    pub const BITMASK_LEN: usize = (LANES + 7) / 8;
}

/// Statically guarantees that a lane count is marked as supported.
///
/// This trait is *sealed*: the list of implementors below is total.
/// Users do not have the ability to mark additional `LaneCount<N>` values as supported.
/// Only SIMD vectors with supported lane counts are constructable.
pub trait SupportedLaneCount: Sealed {
    #[doc(hidden)]
    type BitMask: Copy + Default + AsRef<[u8]> + AsMut<[u8]>;
}

impl<const LANES: usize> Sealed for LaneCount<LANES> {}

impl SupportedLaneCount for LaneCount<1> {
    type BitMask = [u8; 1];
}
impl SupportedLaneCount for LaneCount<2> {
    type BitMask = [u8; 1];
}
impl SupportedLaneCount for LaneCount<4> {
    type BitMask = [u8; 1];
}
impl SupportedLaneCount for LaneCount<8> {
    type BitMask = [u8; 1];
}
impl SupportedLaneCount for LaneCount<16> {
    type BitMask = [u8; 2];
}
impl SupportedLaneCount for LaneCount<32> {
    type BitMask = [u8; 4];
}
impl SupportedLaneCount for LaneCount<64> {
    type BitMask = [u8; 8];
}
