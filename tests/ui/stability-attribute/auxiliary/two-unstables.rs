#![stable(feature = "stable_feature", since = "1.0.0")]
#![feature(staged_api)]
#![crate_type = "lib"]

#[unstable(feature = "a", issue = "1", reason = "reason")]
#[unstable(feature = "b", issue = "2")]
pub struct Foo;

#[stable(feature = "stable_feature", since = "1.0.0")]
#[rustc_const_unstable(feature = "c", issue = "3", reason = "reason")]
#[rustc_const_unstable(feature = "d", issue = "4")]
pub const fn nothing() {}

#[stable(feature = "stable_feature", since = "1.0.0")]
pub trait Trait {
    #[stable(feature = "stable_feature", since = "1.0.0")]
    #[rustc_default_body_unstable(feature = "e", issue = "5", reason = "reason")]
    #[rustc_default_body_unstable(feature = "f", issue = "6")]
    fn method() {}
}

#[unstable(feature = "g", issue = "7", reason = "reason")]
#[unstable(feature = "h", issue = "8")]
#[macro_export]
macro_rules! mac {
    () => ()
}
