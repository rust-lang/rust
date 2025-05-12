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

#[derive(Default)]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[non_exhaustive]
pub struct UnstableStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub stable: bool,
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub stable2: usize,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable: u8,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[non_exhaustive]
pub struct OnlyUnstableStruct {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable: u8,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable2: bool,
}

impl OnlyUnstableStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub fn new() -> Self {
        Self {
            unstable: 0,
            unstable2: false,
        }
    }
}
