#![doc(hidden)]

macro_rules! uint_module {
    ($T:ident, $Max:expr) => (
        uint_module!(
            $T,
            $Max,
            #[stable(feature = "rust1", since = "1.0.0")]
        );
    );
    ($T:ident, $Max:expr, #[$attr:meta]) => (
        doc_comment! {
            "The smallest value that can be represented by this integer type, which is 0.",
            #[$attr]
            pub const MIN: $T = $T::min_value();
        }
        doc_comment! {
            concat!(
                "The largest value that can be represented by this integer type, which is ",
                stringify!($Max),
                "."
            ),
            #[$attr]
            pub const MAX: $T = $T::max_value();
        }
    )
}
