#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Foo {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Stable,
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Stable2,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Unstable,
}
