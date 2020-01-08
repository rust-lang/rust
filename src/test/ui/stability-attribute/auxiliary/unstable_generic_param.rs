#![crate_type = "lib"]
#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait<#[unstable(feature = "ty", issue = "none")] T = ()> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}
