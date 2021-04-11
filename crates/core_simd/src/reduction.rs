macro_rules! impl_integer_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::LanesAtMost32
        {
            /// Horizontal wrapping add.  Computes the sum of the lanes of the vector, with wrapping addition.
            #[inline]
            pub fn wrapping_sum(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0) }
            }

            /// Horizontal wrapping multiply.  Computes the product of the lanes of the vector, with wrapping multiplication.
            #[inline]
            pub fn wrapping_product(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1) }
            }

            /// Horizontal bitwise "and".  Computes the cumulative bitwise "and" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_and(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_and(self) }
            }

            /// Horizontal bitwise "or".  Computes the cumulative bitwise "or" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_or(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_or(self) }
            }

            /// Horizontal bitwise "xor".  Computes the cumulative bitwise "xor" across the lanes of
            /// the vector.
            #[inline]
            pub fn horizontal_xor(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_xor(self) }
            }

            /// Horizontal maximum.  Computes the maximum lane in the vector.
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Computes the minimum lane in the vector.
            #[inline]
            pub fn horizontal_min(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_min(self) }
            }
        }
    }
}

macro_rules! impl_float_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::LanesAtMost32
        {

            /// Horizontal add.  Computes the sum of the lanes of the vector.
            #[inline]
            pub fn sum(self) -> $scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_slice().iter().sum()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0.) }
                }
            }

            /// Horizontal multiply.  Computes the sum of the lanes of the vector.
            #[inline]
            pub fn product(self) -> $scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_slice().iter().product()
                } else {
                    unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1.) }
                }
            }

            /// Horizontal maximum.  Computes the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn horizontal_max(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Horizontal minimum.  Computes the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn horizontal_min(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_min(self) }
            }
        }
    }
}

macro_rules! impl_full_mask_reductions {
    { $name:ident, $inner:ident } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            crate::$inner<LANES>: crate::LanesAtMost32
        {
            /// Returns true if any lane is set, or false otherwise.
            #[inline]
            pub fn any(self) -> bool {
                unsafe { crate::intrinsics::simd_reduce_any(self.to_int()) }
            }

            /// Returns true if all lanes are set, or false otherwise.
            #[inline]
            pub fn all(self) -> bool {
                unsafe { crate::intrinsics::simd_reduce_all(self.to_int()) }
            }
        }
    }
}

macro_rules! impl_opaque_mask_reductions {
    { $name:ident, $inner:ident, $bits_ty:ident } => {
        impl<const LANES: usize> $name<LANES>
        where
            $bits_ty<LANES>: crate::LanesAtMost32
        {
            /// Returns true if any lane is set, or false otherwise.
            #[inline]
            pub fn any(self) -> bool {
                self.0.any()
            }

            /// Returns true if all lanes are set, or false otherwise.
            #[inline]
            pub fn all(self) -> bool {
                self.0.all()
            }
        }
    }
}

impl<const LANES: usize> crate::BitMask<LANES>
where
    crate::BitMask<LANES>: crate::LanesAtMost32,
{
    /// Returns true if any lane is set, or false otherwise.
    #[inline]
    pub fn any(self) -> bool {
        self != Self::splat(false)
    }

    /// Returns true if all lanes are set, or false otherwise.
    #[inline]
    pub fn all(self) -> bool {
        self == Self::splat(true)
    }
}
