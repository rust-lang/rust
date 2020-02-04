#![doc(hidden)]

macro_rules! doc_comment {
    ($x:expr, $($tt:tt)*) => {
        #[doc = $x]
        $($tt)*
    };
}

macro_rules! uint_module {
    ($T:ident) => (uint_module!($T, #[stable(feature = "rust1", since = "1.0.0")]););
    ($T:ident, #[$attr:meta]) => (
        doc_comment! {
            concat!("**This method is soft-deprecated.**

            Although using it won’t cause compilation warning,
            new code should use [`", stringify!($T), "::MIN", "`] instead.

            The smallest value that can be represented by this integer type."),
            #[$attr]
            pub const MIN: $T = $T::min_value();
        }

        doc_comment! {
            concat!("**This method is soft-deprecated.**

            Although using it won’t cause compilation warning,
            new code should use [`", stringify!($T), "::MAX", "`] instead.

            The largest value that can be represented by this integer type."),
            #[$attr]
            pub const MAX: $T = $T::max_value();
        }
    )
}
