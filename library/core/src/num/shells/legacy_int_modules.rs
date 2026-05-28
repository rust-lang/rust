#![doc(hidden)]

macro_rules! legacy_int_module {
    ($T:ident) => (legacy_int_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        #[$attr]
        #[deprecated(
            since = "TBD",
            note = "all constants in this module replaced by associated constants on the type"
        )]
        #[rustc_diagnostic_item = concat!(stringify!($T), "_legacy_mod")]
        pub mod $T {
            #![doc = concat!("Redundant constants module for the [`", stringify!($T), "` primitive type][", stringify!($T), "].")]
            //!
            //! New code should use the associated constants directly on the primitive type.

            #[doc = concat!(
                "The smallest value that can be represented by this integer type. Use ",
                "[`", stringify!($T), "::MIN", "`] instead."
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
            #[deprecated(since = "TBD", note = "replaced by the `MIN` associated constant on this type")]
            #[rustc_diagnostic_item = concat!(stringify!($T), "_legacy_const_min")]
            pub const MIN: $T = $T::MIN;

            #[doc = concat!(
                "The largest value that can be represented by this integer type. Use ",
                "[`", stringify!($T), "::MAX", "`] instead."
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
            #[deprecated(since = "TBD", note = "replaced by the `MAX` associated constant on this type")]
            #[rustc_diagnostic_item = concat!(stringify!($T), "_legacy_const_max")]
            pub const MAX: $T = $T::MAX;
        }
    )
}

legacy_int_module! { i128, #[stable(feature = "i128", since = "1.26.0")] }
legacy_int_module! { i16 }
legacy_int_module! { i32 }
legacy_int_module! { i64 }
legacy_int_module! { i8 }
legacy_int_module! { isize }
legacy_int_module! { u128, #[stable(feature = "i128", since = "1.26.0")] }
legacy_int_module! { u16 }
legacy_int_module! { u32 }
legacy_int_module! { u64 }
legacy_int_module! { u8 }
legacy_int_module! { usize }
