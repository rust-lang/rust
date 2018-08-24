#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

/// docs for my_macro
#[unstable(feature = "macro_test", issue = "0")]
#[rustc_deprecated(since = "1.2.3", reason = "text")]
#[macro_export]
macro_rules! my_macro {
    () => ()
}
