#![doc(hidden)]

macro_rules! int_module {
    ($T:ident) => (int_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        /// The smallest value that can be represented by this integer type.
        #[$attr]
        #[rustc_deprecated(since = "1.42.0", reason = "replaced by associated constant MIN")]
        pub const MIN: $T = $T::MIN;
        /// The largest value that can be represented by this integer type.
        #[$attr]
        #[rustc_deprecated(since = "1.42.0", reason = "replaced by associated constant MAX")]
        pub const MAX: $T = $T::MAX;
    )
}
