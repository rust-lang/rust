#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "1.0")]

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct Unstable {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable: u8,
}

#[stable(feature = "stable_test_feature", since = "1.0")]
pub struct Stable {
    #[stable(feature = "stable_test_feature", since = "1.0")]
    pub stable: u8,
}

#[stable(feature = "stable_test_feature", since = "1.0")]
pub struct StableWithUnstableField {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable: u8,
}

#[stable(feature = "stable_test_feature", since = "1.0")]
pub struct StableWithUnstableFieldType {
    #[stable(feature = "stable_test_feature", since = "1.0")]
    pub stable: Unstable,
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct UnstableWithStableFieldType {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub unstable: Stable,
}
