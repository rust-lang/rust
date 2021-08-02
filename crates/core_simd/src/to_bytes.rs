macro_rules! impl_to_bytes {
    { $name:ident, $size:literal } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
            crate::LaneCount<{{ $size * LANES }}>: crate::SupportedLaneCount,
        {
            /// Return the memory representation of this integer as a byte array in native byte
            /// order.
            pub fn to_ne_bytes(self) -> crate::SimdU8<{{ $size * LANES }}> {
                unsafe { core::mem::transmute_copy(&self) }
            }

            /// Create a native endian integer value from its memory representation as a byte array
            /// in native endianness.
            pub fn from_ne_bytes(bytes: crate::SimdU8<{{ $size * LANES }}>) -> Self {
                unsafe { core::mem::transmute_copy(&bytes) }
            }
        }
    }
}

impl_to_bytes! { SimdU8, 1 }
impl_to_bytes! { SimdU16, 2 }
impl_to_bytes! { SimdU32, 4 }
impl_to_bytes! { SimdU64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { SimdUsize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { SimdUsize, 8 }

impl_to_bytes! { SimdI8, 1 }
impl_to_bytes! { SimdI16, 2 }
impl_to_bytes! { SimdI32, 4 }
impl_to_bytes! { SimdI64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { SimdIsize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { SimdIsize, 8 }
