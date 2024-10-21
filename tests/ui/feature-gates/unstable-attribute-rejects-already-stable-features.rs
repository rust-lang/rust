//! Ensure #[unstable] doesn't accept already stable features

#![feature(staged_api)]
#![stable(feature = "rust_test", since = "1.0.0")]

#[unstable(feature = "arbitrary_enum_discriminant", issue = "42")] //~ ERROR can't mark as unstable using an already stable feature
#[rustc_const_unstable(feature = "arbitrary_enum_discriminant", issue = "42")] //~ ERROR can't mark as unstable using an already stable feature
const fn my_fun() {}

fn main() {}
