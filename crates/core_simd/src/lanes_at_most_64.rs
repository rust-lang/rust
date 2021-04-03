/// Implemented for bitmask sizes that are supported by the implementation.
pub trait LanesAtMost32 {}

macro_rules! impl_for {
    { $name:ident } => {
        impl LanesAtMost32 for $name<1> {}
        impl LanesAtMost32 for $name<2> {}
        impl LanesAtMost32 for $name<4> {}
        impl LanesAtMost32 for $name<8> {}
        impl LanesAtMost32 for $name<16> {}
        impl LanesAtMost32 for $name<32> {}
    }
}

use crate::*;

impl_for! { SimdU8 }
impl_for! { SimdU16 }
impl_for! { SimdU32 }
impl_for! { SimdU64 }
impl_for! { SimdU128 }
impl_for! { SimdUsize }

impl_for! { SimdI8 }
impl_for! { SimdI16 }
impl_for! { SimdI32 }
impl_for! { SimdI64 }
impl_for! { SimdI128 }
impl_for! { SimdIsize }

impl_for! { SimdF32 }
impl_for! { SimdF64 }

impl_for! { BitMask }
