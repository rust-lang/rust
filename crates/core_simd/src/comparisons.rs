use crate::LanesAtMost32;

macro_rules! implement_mask_ops {
    { $($vector:ident => $mask:ident ($inner_ty:ident),)* } => {
        $(
            impl<const LANES: usize> crate::$vector<LANES>
            where
                crate::$vector<LANES>: LanesAtMost32,
                crate::$inner_ty<LANES>: LanesAtMost32,
                crate::$mask<LANES>: crate::Mask,
            {
                /// Test if each lane is equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_eq(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_eq(self, other))
                    }
                }

                /// Test if each lane is not equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ne(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_ne(self, other))
                    }
                }

                /// Test if each lane is less than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_lt(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_lt(self, other))
                    }
                }

                /// Test if each lane is greater than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_gt(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_gt(self, other))
                    }
                }

                /// Test if each lane is less than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_le(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_le(self, other))
                    }
                }

                /// Test if each lane is greater than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ge(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$mask::from_int_unchecked(crate::intrinsics::simd_ge(self, other))
                    }
                }
            }
        )*
    }
}

implement_mask_ops! {
    SimdI8 => Mask8 (SimdI8),
    SimdI16 => Mask16 (SimdI16),
    SimdI32 => Mask32 (SimdI32),
    SimdI64 => Mask64 (SimdI64),
    SimdIsize => MaskSize (SimdIsize),

    SimdU8 => Mask8 (SimdI8),
    SimdU16 => Mask16 (SimdI16),
    SimdU32 => Mask32 (SimdI32),
    SimdU64 => Mask64 (SimdI64),
    SimdUsize => MaskSize (SimdIsize),

    SimdF32 => Mask32 (SimdI32),
    SimdF64 => Mask64 (SimdI64),
}
