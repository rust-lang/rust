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
                unsafe { core::mem::transmute_copy(&self) }
            }

            /// Create a native endian integer value from its memory representation as a byte array
            /// in native endianness.
            pub fn from_ne_bytes(bytes: crate::simd::Simd<u8, {{ $size * LANES }}>) -> Self {
                unsafe { core::mem::transmute_copy(&bytes) }
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
