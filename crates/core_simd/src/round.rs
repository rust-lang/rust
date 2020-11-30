macro_rules! implement {
    {
        impl $type:ident {
            int_type = $int_type:ident
        }
    } => {
        mod $type {
            impl crate::$type {
                /// Returns the largest integer less than or equal to each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn floor(self) -> Self {
                    unsafe { crate::intrinsics::simd_floor(self) }
                }

                /// Returns the smallest integer greater than or equal to each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn ceil(self) -> Self {
                    unsafe { crate::intrinsics::simd_ceil(self) }
                }

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
                pub unsafe fn to_int_unchecked(self) -> crate::$int_type {
                    crate::intrinsics::simd_cast(self)
                }

                /// Creates a floating-point vector from an integer vector.  Rounds values that are
                /// not exactly representable.
                #[inline]
                pub fn round_from_int(value: crate::$int_type) -> Self {
                    unsafe { crate::intrinsics::simd_cast(value) }
                }
            }
        }
    }
}

implement! {
    impl f32x2 {
        int_type = i32x2
    }
}

implement! {
    impl f32x4 {
        int_type = i32x4
    }
}

implement! {
    impl f32x8 {
        int_type = i32x8
    }
}

implement! {
    impl f32x16 {
        int_type = i32x16
    }
}

implement! {
    impl f64x2 {
        int_type = i64x2
    }
}

implement! {
    impl f64x4 {
        int_type = i64x4
    }
}

implement! {
    impl f64x8 {
        int_type = i64x8
    }
}
