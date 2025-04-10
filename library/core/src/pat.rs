//! Helper module for exporting the `pattern_type` macro

/// Creates a pattern type.
/// ```ignore (cannot test this from within core yet)
/// type Positive = std::pat::pattern_type!(i32 is 1..);
/// ```
#[macro_export]
#[rustc_builtin_macro(pattern_type)]
#[unstable(feature = "pattern_type_macro", issue = "123646")]
macro_rules! pattern_type {
    ($($arg:tt)*) => {
        /* compiler built-in */
    };
}

/// A trait implemented for integer types and `char`.
/// Useful in the future for generic pattern types, but
/// used right now to simplify ast lowering of pattern type ranges.
#[unstable(feature = "pattern_type_range_trait", issue = "123646")]
#[rustc_const_unstable(feature = "pattern_type_range_trait", issue = "123646")]
#[const_trait]
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid base type for range patterns",
    label = "only integer types and `char` are supported"
)]
pub trait RangePattern {
    /// Trait version of the inherent `MIN` assoc const.
    #[lang = "RangeMin"]
    const MIN: Self;

    /// Trait version of the inherent `MIN` assoc const.
    #[lang = "RangeMax"]
    const MAX: Self;

    /// A compile-time helper to subtract 1 for exclusive ranges.
    #[lang = "RangeSub"]
    #[track_caller]
    fn sub_one(self) -> Self;
}

macro_rules! impl_range_pat {
    ($($ty:ty,)*) => {
        $(
            #[rustc_const_unstable(feature = "pattern_type_range_trait", issue = "123646")]
            impl const RangePattern for $ty {
                const MIN: $ty = <$ty>::MIN;
                const MAX: $ty = <$ty>::MAX;
                fn sub_one(self) -> Self {
                    match self.checked_sub(1) {
                        Some(val) => val,
                        None => panic!("exclusive range end at minimum value of type")
                    }
                }
            }
        )*
    }
}

impl_range_pat! {
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
}

#[rustc_const_unstable(feature = "pattern_type_range_trait", issue = "123646")]
impl const RangePattern for char {
    const MIN: Self = char::MIN;

    const MAX: Self = char::MAX;

    fn sub_one(self) -> Self {
        match char::from_u32(self as u32 - 1) {
            None => panic!("exclusive range to start of valid chars"),
            Some(val) => val,
        }
    }
}
