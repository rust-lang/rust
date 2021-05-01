/// Implemented for vectors that are supported by the implementation.
pub trait LanesAtMost32: sealed::Sealed {
    #[doc(hidden)]
    type BitMask: Into<u64>;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_for {
    { $name:ident } => {
        impl<const LANES: usize> sealed::Sealed for $name<LANES>
        where
            $name<LANES>: LanesAtMost32,
        {}

        impl LanesAtMost32 for $name<1> {
            type BitMask = u8;
        }
        impl LanesAtMost32 for $name<2> {
            type BitMask = u8;
        }
        impl LanesAtMost32 for $name<4> {
            type BitMask = u8;
        }
        impl LanesAtMost32 for $name<8> {
            type BitMask = u8;
        }
        impl LanesAtMost32 for $name<16> {
            type BitMask = u16;
        }
        impl LanesAtMost32 for $name<32> {
            type BitMask = u32;
        }
    }
}

use crate::*;

impl_for! { SimdU8 }
impl_for! { SimdU16 }
impl_for! { SimdU32 }
impl_for! { SimdU64 }
impl_for! { SimdUsize }

impl_for! { SimdI8 }
impl_for! { SimdI16 }
impl_for! { SimdI32 }
impl_for! { SimdI64 }
impl_for! { SimdIsize }

impl_for! { SimdF32 }
impl_for! { SimdF64 }
