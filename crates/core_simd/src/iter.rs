macro_rules! impl_traits {
    { $type:ident } => {
        impl<const LANES: usize> core::iter::Sum<Self> for crate::$type<LANES>
        where
            Self: crate::Vector,
        {
            fn sum<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Add::add)
            }
        }

        impl<const LANES: usize> core::iter::Product<Self> for crate::$type<LANES>
        where
            Self: crate::Vector,
        {
            fn product<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Mul::mul)
            }
        }

        impl<'a, const LANES: usize> core::iter::Sum<&'a Self> for crate::$type<LANES>
        where
            Self: crate::Vector,
        {
            fn sum<I: core::iter::Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Add::add)
            }
        }

        impl<'a, const LANES: usize> core::iter::Product<&'a Self> for crate::$type<LANES>
        where
            Self: crate::Vector,
        {
            fn product<I: core::iter::Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Mul::mul)
            }
        }
    }
}

impl_traits! { SimdF32 }
impl_traits! { SimdF64 }
impl_traits! { SimdU8 }
impl_traits! { SimdU16 }
impl_traits! { SimdU32 }
impl_traits! { SimdU64 }
impl_traits! { SimdUsize }
impl_traits! { SimdI8 }
impl_traits! { SimdI16 }
impl_traits! { SimdI32 }
impl_traits! { SimdI64 }
impl_traits! { SimdIsize }
