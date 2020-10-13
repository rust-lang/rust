macro_rules! implement {
    {
        impl $type:ident {
            int_type = $int_type:ident,
            floor = $floor_intrinsic:literal,
            ceil = $ceil_intrinsic:literal,
            round = $round_intrinsic:literal,
            trunc = $trunc_intrinsic:literal,
        }
    } => {
        mod $type {
            #[allow(improper_ctypes)]
            extern "C" {
                #[link_name = $floor_intrinsic]
                fn floor_intrinsic(x: crate::$type) -> crate::$type;
                #[link_name = $ceil_intrinsic]
                fn ceil_intrinsic(x: crate::$type) -> crate::$type;
                #[link_name = $round_intrinsic]
                fn round_intrinsic(x: crate::$type) -> crate::$type;
                #[link_name = $trunc_intrinsic]
                fn trunc_intrinsic(x: crate::$type) -> crate::$type;
            }

            impl crate::$type {
                /// Returns the largest integer less than or equal to each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn floor(self) -> Self {
                    unsafe { floor_intrinsic(self) }
                }

                /// Returns the smallest integer greater than or equal to each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn ceil(self) -> Self {
                    unsafe { ceil_intrinsic(self) }
                }

                /// Returns the nearest integer to each lane. Round half-way cases away from 0.0.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn round(self) -> Self {
                    unsafe { round_intrinsic(self) }
                }

                /// Returns the integer part of each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn trunc(self) -> Self {
                    unsafe { trunc_intrinsic(self) }
                }

                /// Returns the fractional part of each lane.
                #[must_use = "method returns a new vector and does not mutate the original value"]
                #[inline]
                pub fn fract(self) -> Self {
                    self - self.trunc()
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
        int_type = i32x2,
        floor = "llvm.floor.v2f32",
        ceil = "llvm.ceil.v2f32",
        round = "llvm.round.v2f32",
        trunc = "llvm.trunc.v2f32",
    }
}

implement! {
    impl f32x4 {
        int_type = i32x4,
        floor = "llvm.floor.v4f32",
        ceil = "llvm.ceil.v4f32",
        round = "llvm.round.v4f32",
        trunc = "llvm.trunc.v4f32",
    }
}

implement! {
    impl f32x8 {
        int_type = i32x8,
        floor = "llvm.floor.v8f32",
        ceil = "llvm.ceil.v8f32",
        round = "llvm.round.v8f32",
        trunc = "llvm.trunc.v8f32",
    }
}

implement! {
    impl f32x16 {
        int_type = i32x16,
        floor = "llvm.floor.v16f32",
        ceil = "llvm.ceil.v16f32",
        round = "llvm.round.v16f32",
        trunc = "llvm.trunc.v16f32",
    }
}

implement! {
    impl f64x2 {
        int_type = i64x2,
        floor = "llvm.floor.v2f64",
        ceil = "llvm.ceil.v2f64",
        round = "llvm.round.v2f64",
        trunc = "llvm.trunc.v2f64",
    }
}

implement! {
    impl f64x4 {
        int_type = i64x4,
        floor = "llvm.floor.v4f64",
        ceil = "llvm.ceil.v4f64",
        round = "llvm.round.v4f64",
        trunc = "llvm.trunc.v4f64",
    }
}

implement! {
    impl f64x8 {
        int_type = i64x8,
        floor = "llvm.floor.v8f64",
        ceil = "llvm.ceil.v8f64",
        round = "llvm.round.v8f64",
        trunc = "llvm.trunc.v8f64",
    }
}
