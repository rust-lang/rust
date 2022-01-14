use super::{mask_impl, Mask, MaskElement};

/// Converts masks to and from bitmasks.
///
/// In a bitmask, each bit represents if the corresponding lane in the mask is set.
pub trait ToBitMask<BitMask> {
    /// Converts a mask to a bitmask.
    fn to_bitmask(self) -> BitMask;

    /// Converts a bitmask to a mask.
    fn from_bitmask(bitmask: BitMask) -> Self;
}

macro_rules! impl_integer_intrinsic {
    { $(unsafe impl ToBitMask<$int:ty> for Mask<_, $lanes:literal>)* } => {
        $(
        impl<T: MaskElement> ToBitMask<$int> for Mask<T, $lanes> {
            fn to_bitmask(self) -> $int {
                unsafe { self.0.to_bitmask_intrinsic() }
            }

            fn from_bitmask(bitmask: $int) -> Self {
                unsafe { Self(mask_impl::Mask::from_bitmask_intrinsic(bitmask)) }
            }
        }
        )*
    }
}

impl_integer_intrinsic! {
    unsafe impl ToBitMask<u8> for Mask<_, 8>
    unsafe impl ToBitMask<u16> for Mask<_, 16>
    unsafe impl ToBitMask<u32> for Mask<_, 32>
    unsafe impl ToBitMask<u64> for Mask<_, 64>
}

macro_rules! impl_integer_via {
    { $(impl ToBitMask<$int:ty, via $via:ty> for Mask<_, $lanes:literal>)* } => {
        $(
        impl<T: MaskElement> ToBitMask<$int> for Mask<T, $lanes> {
            fn to_bitmask(self) -> $int {
                let bitmask: $via = self.to_bitmask();
                bitmask as _
            }

            fn from_bitmask(bitmask: $int) -> Self {
                Self::from_bitmask(bitmask as $via)
            }
        }
        )*
    }
}

impl_integer_via! {
    impl ToBitMask<u16, via u8> for Mask<_, 8>
    impl ToBitMask<u32, via u8> for Mask<_, 8>
    impl ToBitMask<u64, via u8> for Mask<_, 8>

    impl ToBitMask<u32, via u16> for Mask<_, 16>
    impl ToBitMask<u64, via u16> for Mask<_, 16>

    impl ToBitMask<u64, via u32> for Mask<_, 32>
}

#[cfg(target_pointer_width = "32")]
impl_integer_via! {
    impl ToBitMask<usize, via u8> for Mask<_, 8>
    impl ToBitMask<usize, via u16> for Mask<_, 16>
    impl ToBitMask<usize, via u32> for Mask<_, 32>
}

#[cfg(target_pointer_width = "64")]
impl_integer_via! {
    impl ToBitMask<usize, via u8> for Mask<_, 8>
    impl ToBitMask<usize, via u16> for Mask<_, 16>
    impl ToBitMask<usize, via u32> for Mask<_, 32>
    impl ToBitMask<usize, via u64> for Mask<_, 64>
}
