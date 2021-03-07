macro_rules! implement {
    {
        $type:ident, $int_type:ident
    } => {
        impl<const LANES: usize> crate::$type<LANES>
        where
            Self: crate::LanesAtMost64,
        {
            /// Returns the largest integer less than or equal to each lane.
            #[cfg(feature = "std")]
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn floor(self) -> Self {
                unsafe { crate::intrinsics::simd_floor(self) }
            }

            /// Returns the smallest integer greater than or equal to each lane.
            #[cfg(feature = "std")]
            #[must_use = "method returns a new vector and does not mutate the original value"]
            #[inline]
            pub fn ceil(self) -> Self {
                unsafe { crate::intrinsics::simd_ceil(self) }
            }
        }

        impl<const LANES: usize> crate::$type<LANES>
        where
            Self: crate::LanesAtMost64,
            crate::$int_type<LANES>: crate::LanesAtMost64,
        {
            /// Rounds toward zero and converts to the same-width integer type, assuming that
            /// the value is finite and fits in that type.
            ///
            /// # Safety
            /// The value must:
            ///
            /// * Not be NaN
            /// * Not be infinite
            /// * Be representable in the return type, after truncating off its fractional part
            #[inline]
            pub unsafe fn to_int_unchecked(self) -> crate::$int_type<LANES> {
                crate::intrinsics::simd_cast(self)
            }

            /// Creates a floating-point vector from an integer vector.  Rounds values that are
            /// not exactly representable.
            #[inline]
            pub fn round_from_int(value: crate::$int_type<LANES>) -> Self {
                unsafe { crate::intrinsics::simd_cast(value) }
            }
        }
    }
}

implement! { SimdF32, SimdI32 }
implement! { SimdF64, SimdI64 }
