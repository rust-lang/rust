#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[non_exhaustive]
pub enum UnstableEnum {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Stable,
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Stable2,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Unstable,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[non_exhaustive]
pub enum OnlyUnstableEnum {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Unstable,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Unstable2,
}

impl OnlyUnstableEnum {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub fn new() -> Self {
        Self::Unstable
    }
}
