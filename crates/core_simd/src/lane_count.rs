mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// A type representing a vector lane count.
pub struct LaneCount<const LANES: usize>;

/// Helper trait for vector lane counts.
pub trait SupportedLaneCount: Sealed {
    /// The bitmask representation of a mask.
    type BitMask: Copy + Default + AsRef<[u8]> + AsMut<[u8]>;

    #[doc(hidden)]
    type IntBitMask;
}

impl<const LANES: usize> Sealed for LaneCount<LANES> {}

impl SupportedLaneCount for LaneCount<1> {
    type BitMask = [u8; 1];
    type IntBitMask = u8;
}
impl SupportedLaneCount for LaneCount<2> {
    type BitMask = [u8; 1];
    type IntBitMask = u8;
}
impl SupportedLaneCount for LaneCount<4> {
    type BitMask = [u8; 1];
    type IntBitMask = u8;
}
impl SupportedLaneCount for LaneCount<8> {
    type BitMask = [u8; 1];
    type IntBitMask = u8;
}
impl SupportedLaneCount for LaneCount<16> {
    type BitMask = [u8; 2];
    type IntBitMask = u16;
}
impl SupportedLaneCount for LaneCount<32> {
    type BitMask = [u8; 4];
    type IntBitMask = u32;
}
