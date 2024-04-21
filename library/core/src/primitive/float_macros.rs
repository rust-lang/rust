// Note: currently limiting this to what f16/f128 already support (which isn't much).
// f32/f64 share essentially their whole API which should be added here eventually.

macro_rules! float_decl {
    () => {
        /// Returns `true` if this value is NaN.
        fn is_nan(self) -> bool;

        /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
        /// positive sign bit and positive infinity. Note that IEEE 754 doesn't assign any
        /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
        /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
        /// `is_sign_positive` on a NaN might produce an unexpected result in some cases.
        /// See [explanation of NaN as a special value](f32) for more info.
        fn is_sign_positive(self) -> bool;

        /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
        /// negative sign bit and negative infinity. Note that IEEE 754 doesn't assign any
        /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
        /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
        /// `is_sign_negative` on a NaN might produce an unexpected result in some cases.
        /// See [explanation of NaN as a special value](f32) for more info.
        fn is_sign_negative(self) -> bool;
    };
}

macro_rules! float_impl {
    () => {
        #[inline]
        fn is_nan(self) -> bool {
            Self::is_nan(self)
        }

        #[inline]
        fn is_sign_positive(self) -> bool {
            Self::is_sign_positive(self)
        }

        #[inline]
        fn is_sign_negative(self) -> bool {
            Self::is_sign_negative(self)
        }
    };
}
