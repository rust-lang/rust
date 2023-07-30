use crate::simd::SimdUint;

macro_rules! impl_to_bytes {
    { $ty:ty, $size:literal } => {
        impl<const LANES: usize> crate::simd::Simd<$ty, LANES>
        where
            crate::simd::LaneCount<LANES>: crate::simd::SupportedLaneCount,
            crate::simd::LaneCount<{{ $size * LANES }}>: crate::simd::SupportedLaneCount,
        {
            /// Return the memory representation of this integer as a byte array in native byte
            /// order.
            pub fn to_ne_bytes(self) -> crate::simd::Simd<u8, {{ $size * LANES }}> {
                // Safety: transmuting between vectors is safe
                unsafe { core::mem::transmute_copy(&self) }
            }

            /// Return the memory representation of this integer as a byte array in big-endian
            /// (network) byte order.
            pub fn to_be_bytes(self) -> crate::simd::Simd<u8, {{ $size * LANES }}> {
                let bytes = self.to_ne_bytes();
                if cfg!(target_endian = "big") {
                    bytes
                } else {
                    bytes.swap_bytes()
                }
            }

            /// Return the memory representation of this integer as a byte array in little-endian
            /// byte order.
            pub fn to_le_bytes(self) -> crate::simd::Simd<u8, {{ $size * LANES }}> {
                let bytes = self.to_ne_bytes();
                if cfg!(target_endian = "little") {
                    bytes
                } else {
                    bytes.swap_bytes()
                }
            }

            /// Create a native endian integer value from its memory representation as a byte array
            /// in native endianness.
            pub fn from_ne_bytes(bytes: crate::simd::Simd<u8, {{ $size * LANES }}>) -> Self {
                // Safety: transmuting between vectors is safe
                unsafe { core::mem::transmute_copy(&bytes) }
            }

            /// Create an integer value from its representation as a byte array in big endian.
            pub fn from_be_bytes(bytes: crate::simd::Simd<u8, {{ $size * LANES }}>) -> Self {
                let bytes = if cfg!(target_endian = "big") {
                    bytes
                } else {
                    bytes.swap_bytes()
                };
                Self::from_ne_bytes(bytes)
            }

            /// Create an integer value from its representation as a byte array in little endian.
            pub fn from_le_bytes(bytes: crate::simd::Simd<u8, {{ $size * LANES }}>) -> Self {
                let bytes = if cfg!(target_endian = "little") {
                    bytes
                } else {
                    bytes.swap_bytes()
                };
                Self::from_ne_bytes(bytes)
            }
        }
    }
}

impl_to_bytes! { u8, 1 }
impl_to_bytes! { u16, 2 }
impl_to_bytes! { u32, 4 }
impl_to_bytes! { u64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { usize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { usize, 8 }

impl_to_bytes! { i8, 1 }
impl_to_bytes! { i16, 2 }
impl_to_bytes! { i32, 4 }
impl_to_bytes! { i64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { isize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { isize, 8 }
