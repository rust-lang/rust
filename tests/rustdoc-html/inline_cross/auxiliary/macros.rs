#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

/// docs for my_macro
#[unstable(feature = "macro_test", issue = "none")]
#[deprecated(since = "1.2.3", note = "text")]
#[macro_export]
macro_rules! my_macro {
    () => {};
}
