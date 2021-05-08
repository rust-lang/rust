mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Supporting trait for vector `select` function
pub trait Select<Mask>: Sealed {
    #[doc(hidden)]
    fn select(mask: Mask, true_values: Self, false_values: Self) -> Self;
}

macro_rules! impl_select {
    {
        $mask:ident ($bits_ty:ident): $($type:ident),*
    } => {
        $(
        impl<const LANES: usize> Sealed for crate::$type<LANES> where Self: crate::LanesAtMost32 {}
        impl<const LANES: usize> Select<crate::$mask<LANES>> for crate::$type<LANES>
        where
            crate::$mask<LANES>: crate::Mask,
            crate::$bits_ty<LANES>: crate::LanesAtMost32,
            Self: crate::LanesAtMost32,
        {
            #[doc(hidden)]
            #[inline]
            fn select(mask: crate::$mask<LANES>, true_values: Self, false_values: Self) -> Self {
                unsafe { crate::intrinsics::simd_select(mask.to_int(), true_values, false_values) }
            }
        }
        )*

        impl<const LANES: usize> Sealed for crate::$mask<LANES>
        where
            Self: crate::Mask,
            crate::$bits_ty<LANES>: crate::LanesAtMost32,
        {}
        impl<const LANES: usize> Select<Self> for crate::$mask<LANES>
        where
            Self: crate::Mask,
            crate::$bits_ty<LANES>: crate::LanesAtMost32,
        {
            #[doc(hidden)]
            #[inline]
            fn select(mask: Self, true_values: Self, false_values: Self) -> Self {
                mask & true_values | !mask & false_values
            }
        }

        impl<const LANES: usize> crate::$mask<LANES>
        where
            Self: crate::Mask,
            crate::$bits_ty<LANES>: crate::LanesAtMost32,
        {
            /// Choose lanes from two vectors.
            ///
            /// For each lane in the mask, choose the corresponding lane from `true_values` if
            /// that lane mask is true, and `false_values` if that lane mask is false.
            ///
            /// ```
            /// # use core_simd::{Mask32, SimdI32};
            /// let a = SimdI32::from_array([0, 1, 2, 3]);
            /// let b = SimdI32::from_array([4, 5, 6, 7]);
            /// let mask = Mask32::from_array([true, false, false, true]);
            /// let c = mask.select(a, b);
            /// assert_eq!(c.to_array(), [0, 5, 6, 3]);
            /// ```
            ///
            /// `select` can also be used with masks:
            /// ```
            /// # use core_simd::{Mask32};
            /// let a = Mask32::from_array([true, true, false, false]);
            /// let b = Mask32::from_array([false, false, true, true]);
            /// let mask = Mask32::from_array([true, false, false, true]);
            /// let c = mask.select(a, b);
            /// assert_eq!(c.to_array(), [true, false, true, false]);
            /// ```
            #[inline]
            pub fn select<S: Select<Self>>(self, true_values: S, false_values: S) -> S {
                S::select(self, true_values, false_values)
            }
        }
    }
}

impl_select! { Mask8 (SimdI8): SimdU8, SimdI8 }
impl_select! { Mask16 (SimdI16): SimdU16, SimdI16 }
impl_select! { Mask32 (SimdI32): SimdU32, SimdI32, SimdF32}
impl_select! { Mask64 (SimdI64): SimdU64, SimdI64, SimdF64}
impl_select! { MaskSize (SimdIsize): SimdUsize, SimdIsize }
