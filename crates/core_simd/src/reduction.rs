macro_rules! impl_integer_reductions {
    { $name:ident, $scalar:ty } => {
        impl<const LANES: usize> crate::$name<LANES>
        where
            Self: crate::LanesAtMost32
        {
            /// Produces the sum of the lanes of the vector, with wrapping addition.
            #[inline]
            pub fn wrapping_sum(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0) }
            }

            /// Produces the sum of the lanes of the vector, with wrapping multiplication.
            #[inline]
            pub fn wrapping_product(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1) }
            }

            /// Sequentially performs bitwise "and" between the lanes of the vector.
            #[inline]
            pub fn and_lanes(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_and(self) }
            }

            /// Sequentially performs bitwise "or" between the lanes of the vector.
            #[inline]
            pub fn or_lanes(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_or(self) }
            }

            /// Sequentially performs bitwise "xor" between the lanes of the vector.
            #[inline]
            pub fn xor_lanes(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_xor(self) }
            }

            /// Returns the maximum lane in the vector.
            #[inline]
            pub fn max_lane(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Returns the minimum lane in the vector.
            #[inline]
            pub fn min_lane(self) -> $scalar {
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

            /// Produces the sum of the lanes of the vector.
            #[inline]
            pub fn sum(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_add_ordered(self, 0.) }
            }

            /// Produces the sum of the lanes of the vector.
            #[inline]
            pub fn product(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_mul_ordered(self, 1.) }
            }

            /// Returns the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn max_lane(self) -> $scalar {
                unsafe { crate::intrinsics::simd_reduce_max(self) }
            }

            /// Returns the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.  This function will not return `NaN` unless all lanes are `NaN`.
            #[inline]
            pub fn min_lane(self) -> $scalar {
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
