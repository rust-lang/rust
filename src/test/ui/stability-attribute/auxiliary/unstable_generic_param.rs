#![crate_type = "lib"]
#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait1<#[unstable(feature = "unstable_default", issue = "none")] T = ()> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait2<T = ()> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}
