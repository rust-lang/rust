#![doc(hidden)]

macro_rules! int_module {
    ($T:ident) => (int_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        /// The smallest value that can be represented by this integer type.
        #[$attr]
        pub const MIN: $T = $T::min_value();
        /// The largest value that can be represented by this integer type.
        #[$attr]
        pub const MAX: $T = $T::max_value();
    )
}
