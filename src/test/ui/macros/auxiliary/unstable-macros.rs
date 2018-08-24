#![feature(staged_api)]
#![stable(feature = "unit_test", since = "1.0.0")]

#[unstable(feature = "unstable_macros", issue = "0")]
#[macro_export]
macro_rules! unstable_macro{ () => () }
