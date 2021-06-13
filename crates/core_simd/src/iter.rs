macro_rules! impl_traits {
    { $type:ident, $scalar:ty } => {
        impl<const LANES: usize> core::iter::Sum<Self> for crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
        {
            fn sum<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Add::add)
            }
        }

        impl<const LANES: usize> core::iter::Product<Self> for crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
        {
            fn product<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Mul::mul)
            }
        }

        impl<const LANES: usize> core::iter::Sum<crate::$type<LANES>> for $scalar
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn sum<I: core::iter::Iterator<Item = crate::$type<LANES>>>(iter: I) -> Self {
                iter.sum::<crate::$type<LANES>>().horizontal_sum()
            }
        }

        impl<const LANES: usize> core::iter::Product<crate::$type<LANES>> for $scalar
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn product<I: core::iter::Iterator<Item = crate::$type<LANES>>>(iter: I) -> Self {
                iter.product::<crate::$type<LANES>>().horizontal_product()
            }
        }

        impl<'a, const LANES: usize> core::iter::Sum<&'a Self> for crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
        {
            fn sum<I: core::iter::Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Add::add)
            }
        }

        impl<'a, const LANES: usize> core::iter::Product<&'a Self> for crate::$type<LANES>
        where
            Self: crate::LanesAtMost32,
        {
            fn product<I: core::iter::Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Default::default(), core::ops::Mul::mul)
            }
        }

        impl<'a, const LANES: usize> core::iter::Sum<&'a crate::$type<LANES>> for $scalar
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn sum<I: core::iter::Iterator<Item = &'a crate::$type<LANES>>>(iter: I) -> Self {
                iter.sum::<crate::$type<LANES>>().horizontal_sum()
            }
        }

        impl<'a, const LANES: usize> core::iter::Product<&'a crate::$type<LANES>> for $scalar
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn product<I: core::iter::Iterator<Item = &'a crate::$type<LANES>>>(iter: I) -> Self {
                iter.product::<crate::$type<LANES>>().horizontal_product()
            }
        }
    }
}

impl_traits! { SimdF32, f32 }
impl_traits! { SimdF64, f64 }
impl_traits! { SimdU8, u8 }
impl_traits! { SimdU16, u16 }
impl_traits! { SimdU32, u32 }
impl_traits! { SimdU64, u64 }
impl_traits! { SimdUsize, usize }
impl_traits! { SimdI8, i8 }
impl_traits! { SimdI16, i16 }
impl_traits! { SimdI32, i32 }
impl_traits! { SimdI64, i64 }
impl_traits! { SimdIsize, isize }
