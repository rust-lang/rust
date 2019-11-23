#![doc(hidden)]

macro_rules! int_module {
    ($T:ident, $Min:expr, $Max:expr) => (
        int_module!(
            $T,
            $Min,
            $Max,
            #[stable(feature = "rust1", since = "1.0.0")]
        );
    );
    ($T:ident, $Min:expr, $Max:expr, #[$attr:meta]) => (
        doc_comment! {
            concat!(
                "The smallest value that can be represented by this integer type, which is ",
                stringify!($Min),
                "."
            ),
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
