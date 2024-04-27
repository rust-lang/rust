#![feature(decl_macro)]
#![feature(staged_api)]
#![stable(feature = "unit_test", since = "1.0.0")]

#[unstable(feature = "unstable_macros", issue = "none")]
#[macro_export]
macro_rules! unstable_macro{ () => () }

#[stable(feature = "deprecated_macros", since = "1.0.0")]
#[deprecated(since = "1.0.0", note = "deprecation note")]
#[macro_export]
macro_rules! deprecated_macro{ () => () }

// FIXME: Cannot use a `pub` macro 2.0 in a staged API crate due to reachability issues.
// #[unstable(feature = "unstable_macros", issue = "none")]
// pub macro unstable_macro_modern() {}
