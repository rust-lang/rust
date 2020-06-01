#![doc(hidden)]

macro_rules! int_module {
    ($T:ident) => (int_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        /// The smallest value that can be represented by this integer type.
        #[$attr]
        #[rustc_deprecated(
            since = "1.46.0",
            reason = "The associated constant MIN is now prefered",
        )]
        pub const MIN: $T = $T::MIN;

        /// The largest value that can be represented by this integer type.
        #[$attr]
        #[rustc_deprecated(
            since = "1.46.0",
            reason = "The associated constant MAX is now prefered",
        )]
        pub const MAX: $T = $T::MAX;
    )
}
