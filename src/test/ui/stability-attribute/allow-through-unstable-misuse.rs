#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![stable(feature = "foo", since = "1.0.0")]

#[unstable(feature = "bar", issue = "none")]
#[rustc_allowed_through_unstable_modules]
pub struct UnstableType(());
