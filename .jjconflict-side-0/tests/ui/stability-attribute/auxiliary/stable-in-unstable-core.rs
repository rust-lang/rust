#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.2.0")]

#[unstable(feature = "unstable_test_feature", issue = "1")]
pub mod new_unstable_module {
    #[stable(feature = "stable_test_feature", since = "1.2.0")]
    pub trait OldTrait {}
}
