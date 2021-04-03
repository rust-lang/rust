use crate::LanesAtMost32;

macro_rules! implement_mask_ops {
    { $($vector:ident => $mask:ident ($inner_mask_ty:ident, $inner_ty:ident),)* } => {
        $(
            impl<const LANES: usize> crate::$vector<LANES>
            where
                crate::$vector<LANES>: LanesAtMost32,
                crate::$inner_ty<LANES>: LanesAtMost32,
            {
                /// Test if each lane is equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_eq(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_eq(self, other))
                            .into()
                    }
                }

                /// Test if each lane is not equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ne(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_ne(self, other))
                            .into()
                    }
                }

                /// Test if each lane is less than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_lt(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_lt(self, other))
                            .into()
                    }
                }

                /// Test if each lane is greater than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_gt(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_gt(self, other))
                            .into()
                    }
                }

                /// Test if each lane is less than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_le(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_le(self, other))
                            .into()
                    }
                }

                /// Test if each lane is greater than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ge(self, other: Self) -> crate::$mask<LANES> {
                    unsafe {
                        crate::$inner_mask_ty::from_int_unchecked(crate::intrinsics::simd_ge(self, other))
                            .into()
                    }
                }
            }
        )*
    }
}

implement_mask_ops! {
    SimdI8 => Mask8 (SimdMask8, SimdI8),
    SimdI16 => Mask16 (SimdMask16, SimdI16),
    SimdI32 => Mask32 (SimdMask32, SimdI32),
    SimdI64 => Mask64 (SimdMask64, SimdI64),
    SimdI128 => Mask128 (SimdMask128, SimdI128),
    SimdIsize => MaskSize (SimdMaskSize, SimdIsize),

    SimdU8 => Mask8 (SimdMask8, SimdI8),
    SimdU16 => Mask16 (SimdMask16, SimdI16),
    SimdU32 => Mask32 (SimdMask32, SimdI32),
    SimdU64 => Mask64 (SimdMask64, SimdI64),
    SimdU128 => Mask128 (SimdMask128, SimdI128),
    SimdUsize => MaskSize (SimdMaskSize, SimdIsize),

    SimdF32 => Mask32 (SimdMask32, SimdI32),
    SimdF64 => Mask64 (SimdMask64, SimdI64),
}
