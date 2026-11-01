// This would belong to `rustc_data_structures`, but `rustc_serialize` needs it too.

/// Addition, but only overflow checked when `cfg(debug_assertions)` is set
/// instead of respecting `-Coverflow-checks`.
///
/// This exists for performance reasons, as we ship rustc with overflow checks.
/// While overflow checks are perf neutral in almost all of the compiler, there
/// are a few particularly hot areas where we don't want overflow checks in our
/// dist builds. Overflow is still a bug there, so we want overflow check for
/// builds with debug assertions.
///
/// That's a long way to say that this should be used in areas where overflow
/// is a bug but overflow checking is too slow.
pub trait DebugStrictAdd {
    /// See [`DebugStrictAdd`].
    fn debug_strict_add(self, other: Self) -> Self;
}

macro_rules! impl_debug_strict_add {
    ($( $ty:ty )*) => {
        $(
            impl DebugStrictAdd for $ty {
                #[inline]
                fn debug_strict_add(self, other: Self) -> Self {
                    if cfg!(debug_assertions) {
                        self + other
                    } else {
                        self.wrapping_add(other)
                    }
                }
            }
        )*
    };
}

/// See [`DebugStrictAdd`].
pub trait DebugStrictSub {
    /// See [`DebugStrictAdd`].
    fn debug_strict_sub(self, other: Self) -> Self;
}

macro_rules! impl_debug_strict_sub {
    ($( $ty:ty )*) => {
        $(
            impl DebugStrictSub for $ty {
                #[inline]
                fn debug_strict_sub(self, other: Self) -> Self {
                    if cfg!(debug_assertions) {
                        self - other
                    } else {
                        self.wrapping_sub(other)
                    }
                }
            }
        )*
    };
}

impl_debug_strict_add! {
    u8 u16 u32 u64 u128 usize
    i8 i16 i32 i64 i128 isize
}

impl_debug_strict_sub! {
    u8 u16 u32 u64 u128 usize
    i8 i16 i32 i64 i128 isize
}
