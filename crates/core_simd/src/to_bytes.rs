use crate::simd::{LaneCount, Simd, SimdElement, SimdFloat, SimdInt, SimdUint, SupportedLaneCount};

mod sealed {
    use super::*;
    pub trait Sealed {}
    impl<T: SimdElement, const N: usize> Sealed for Simd<T, N> where LaneCount<N>: SupportedLaneCount {}
}
use sealed::Sealed;

/// Convert SIMD vectors to vectors of bytes
pub trait ToBytes: Sealed {
    /// This type, reinterpreted as bytes.
    type Bytes;

    /// Return the memory representation of this integer as a byte array in native byte
    /// order.
    fn to_ne_bytes(self) -> Self::Bytes;

    /// Return the memory representation of this integer as a byte array in big-endian
    /// (network) byte order.
    fn to_be_bytes(self) -> Self::Bytes;

    /// Return the memory representation of this integer as a byte array in little-endian
    /// byte order.
    fn to_le_bytes(self) -> Self::Bytes;

    /// Create a native endian integer value from its memory representation as a byte array
    /// in native endianness.
    fn from_ne_bytes(bytes: Self::Bytes) -> Self;

    /// Create an integer value from its representation as a byte array in big endian.
    fn from_be_bytes(bytes: Self::Bytes) -> Self;

    /// Create an integer value from its representation as a byte array in little endian.
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
}

macro_rules! swap_bytes {
    { f32, $x:expr } => { Simd::from_bits($x.to_bits().swap_bytes()) };
    { f64, $x:expr } => { Simd::from_bits($x.to_bits().swap_bytes()) };
    { $ty:ty, $x:expr } => { $x.swap_bytes() }
}

macro_rules! impl_to_bytes {
    { $ty:tt, $size:tt } => {
        impl_to_bytes! { $ty, $size * 1 }
        impl_to_bytes! { $ty, $size * 2 }
        impl_to_bytes! { $ty, $size * 4 }
        impl_to_bytes! { $ty, $size * 8 }
        impl_to_bytes! { $ty, $size * 16 }
        impl_to_bytes! { $ty, $size * 32 }
        impl_to_bytes! { $ty, $size * 64 }
    };

    // multiply element size by number of elements
    { $ty:tt, 1 * $elems:literal } => { impl_to_bytes! { @impl [$ty; $elems], $elems } };
    { $ty:tt, $size:literal * 1 } => { impl_to_bytes! { @impl [$ty; 1], $size } };
    { $ty:tt, 2 * 2  } => { impl_to_bytes! { @impl [$ty; 2], 4  } };
    { $ty:tt, 2 * 4  } => { impl_to_bytes! { @impl [$ty; 4], 8  } };
    { $ty:tt, 2 * 8  } => { impl_to_bytes! { @impl [$ty; 8], 16 } };
    { $ty:tt, 2 * 16 } => { impl_to_bytes! { @impl [$ty; 16], 32 } };
    { $ty:tt, 2 * 32 } => { impl_to_bytes! { @impl [$ty; 32], 64 } };
    { $ty:tt, 4 * 2  } => { impl_to_bytes! { @impl [$ty; 2], 8  } };
    { $ty:tt, 4 * 4  } => { impl_to_bytes! { @impl [$ty; 4], 16 } };
    { $ty:tt, 4 * 8  } => { impl_to_bytes! { @impl [$ty; 8], 32 } };
    { $ty:tt, 4 * 16 } => { impl_to_bytes! { @impl [$ty; 16], 64 } };
    { $ty:tt, 8 * 2  } => { impl_to_bytes! { @impl [$ty; 2], 16 } };
    { $ty:tt, 8 * 4  } => { impl_to_bytes! { @impl [$ty; 4], 32 } };
    { $ty:tt, 8 * 8  } => { impl_to_bytes! { @impl [$ty; 8], 64 } };

    // unsupported number of lanes
    { $ty:ty, $a:literal * $b:literal } => { };

    { @impl [$ty:tt; $elem:literal], $bytes:literal } => {
        impl ToBytes for Simd<$ty, $elem> {
            type Bytes = Simd<u8, $bytes>;

            #[inline]
            fn to_ne_bytes(self) -> Self::Bytes {
                // Safety: transmuting between vectors is safe
                unsafe { core::mem::transmute(self) }
            }

            #[inline]
            fn to_be_bytes(mut self) -> Self::Bytes {
                if !cfg!(target_endian = "big") {
                    self = swap_bytes!($ty, self);
                }
                self.to_ne_bytes()
            }

            #[inline]
            fn to_le_bytes(mut self) -> Self::Bytes {
                if !cfg!(target_endian = "little") {
                    self = swap_bytes!($ty, self);
                }
                self.to_ne_bytes()
            }

            #[inline]
            fn from_ne_bytes(bytes: Self::Bytes) -> Self {
                // Safety: transmuting between vectors is safe
                unsafe { core::mem::transmute(bytes) }
            }

            #[inline]
            fn from_be_bytes(bytes: Self::Bytes) -> Self {
                let ret = Self::from_ne_bytes(bytes);
                if cfg!(target_endian = "big") {
                    ret
                } else {
                    swap_bytes!($ty, ret)
                }
            }

            #[inline]
            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                let ret = Self::from_ne_bytes(bytes);
                if cfg!(target_endian = "little") {
                    ret
                } else {
                    swap_bytes!($ty, ret)
                }
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

impl_to_bytes! { f32, 4 }
impl_to_bytes! { f64, 8 }
