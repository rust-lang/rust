mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Supporting trait for byte conversion functions.
pub trait ToBytes: Sealed {
    /// The bytes representation of this type.
    type Bytes;

    #[doc(hidden)]
    fn to_bytes_impl(self) -> Self::Bytes;

    #[doc(hidden)]
    fn from_bytes_impl(bytes: Self::Bytes) -> Self;
}

macro_rules! impl_to_bytes {
    { $name:ident, $($int_width:literal -> $byte_width:literal),* } => {
        $(
        impl Sealed for crate::$name<$int_width> where Self: crate::LanesAtMost32 {}
        impl ToBytes for crate::$name<$int_width>
        where
            Self: crate::LanesAtMost32,
            crate::SimdU8<$byte_width>: crate::LanesAtMost32,
        {
            type Bytes = crate::SimdU8<$byte_width>;
            fn to_bytes_impl(self) -> Self::Bytes {
                unsafe { core::mem::transmute(self) }
            }
            fn from_bytes_impl(bytes: Self::Bytes) -> Self {
                unsafe { core::mem::transmute(bytes) }
            }
        }
        )*

        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: ToBytes + crate::LanesAtMost32,
        {
            /// Return the memory representation of this integer as a byte array in native byte
            /// order.
            pub fn to_ne_bytes(self) -> <Self as ToBytes>::Bytes { self.to_bytes_impl() }

            /// Create a native endian integer value from its memory representation as a byte array
            /// in native endianness.
            pub fn from_ne_bytes(bytes: <Self as ToBytes>::Bytes) -> Self { Self::from_bytes_impl(bytes) }
        }
    }
}

impl_to_bytes! { SimdU8, 1 -> 1, 2 -> 2, 4 -> 4, 8 -> 8, 16 -> 16, 32 -> 32 }
impl_to_bytes! { SimdU16, 1 -> 2, 2 -> 4, 4 -> 8, 8 -> 16, 16 -> 32 }
impl_to_bytes! { SimdU32, 1 -> 4, 2 -> 8, 4 -> 16, 8 -> 32 }
impl_to_bytes! { SimdU64, 1 -> 8, 2 -> 16, 4 -> 32 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { SimdUsize, 1 -> 4, 2 -> 8, 4 -> 16, 8 -> 32 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { SimdUsize, 1 -> 8, 2 -> 16, 4 -> 32 }

impl_to_bytes! { SimdI8, 1 -> 1, 2 -> 2, 4 -> 4, 8 -> 8, 16 -> 16, 32 -> 32 }
impl_to_bytes! { SimdI16, 1 -> 2, 2 -> 4, 4 -> 8, 8 -> 16, 16 -> 32 }
impl_to_bytes! { SimdI32, 1 -> 4, 2 -> 8, 4 -> 16, 8 -> 32 }
impl_to_bytes! { SimdI64, 1 -> 8, 2 -> 16, 4 -> 32 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { SimdIsize, 1 -> 4, 2 -> 8, 4 -> 16, 8 -> 32 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { SimdIsize, 1 -> 8, 2 -> 16, 4 -> 32 }
