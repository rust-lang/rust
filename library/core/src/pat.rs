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
