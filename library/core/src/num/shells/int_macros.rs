#![doc(hidden)]

macro_rules! int_module {
    ($T:ident) => (int_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        #[doc = concat!(
            "The smallest value that can be represented by this integer type. Use ",
            "[`", stringify!($T), "::MIN", "`](../../std/primitive.", stringify!($T), ".html#associatedconstant.MIN)",
            " instead.",
        )]
        ///
        /// # Examples
        ///
        /// ```rust
        /// // deprecated way
        #[doc = concat!("let min = std::", stringify!($T), "::MIN;")]
        ///
        /// // intended way
        #[doc = concat!("let min = ", stringify!($T), "::MIN;")]
        /// ```
        ///
        #[$attr]
        pub const MIN: $T = $T::MIN;

        #[doc = concat!(
            "The largest value that can be represented by this integer type. Use ",
            "[`", stringify!($T), "::MAX", "`](../../std/primitive.", stringify!($T), ".html#associatedconstant.MAX)",
            " instead.",
        )]
        ///
        /// # Examples
        ///
        /// ```rust
        /// // deprecated way
        #[doc = concat!("let max = std::", stringify!($T), "::MAX;")]
        ///
        /// // intended way
        #[doc = concat!("let max = ", stringify!($T), "::MAX;")]
        /// ```
        ///
        #[$attr]
        pub const MAX: $T = $T::MAX;
    )
}
