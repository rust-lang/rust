#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable {
    pub inherit: u8,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub override1: u8,
    #[deprecated(since = "1.0.0", note = "text")]
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub override2: u8,
    #[stable(feature = "rust2", since = "2.0.0")]
    pub override3: u8,
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable2(#[stable(feature = "rust2", since = "2.0.0")] pub u8,
                   #[unstable(feature = "unstable_test_feature", issue = "none")] pub u8,
                   #[unstable(feature = "unstable_test_feature", issue = "none")]
                   #[deprecated(since = "1.0.0", note = "text")] pub u8,
                   pub u8);

#[stable(feature = "rust1", since = "1.0.0")]
pub enum Stable3 {
    Inherit(u8),
    InheritOverride(#[stable(feature = "rust2", since = "2.0.0")] u8),
    #[stable(feature = "rust2", since = "2.0.0")]
    Override1,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Override2,
    #[deprecated(since = "1.0.0", note = "text")]
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    Override3,
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct Unstable {
    pub inherit: u8,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub override1: u8,
    #[deprecated(since = "1.0.0", note = "text")]
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub override2: u8,
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct Unstable2(pub u8,
                     #[stable(feature = "rust1", since = "1.0.0")] pub u8,
                     #[unstable(feature = "unstable_test_feature", issue = "none")]
                     #[deprecated(since = "1.0.0", note = "text")] pub u8);

#[unstable(feature = "unstable_test_feature", issue = "none")]
#[deprecated(since = "1.0.0", note = "text")]
pub struct Deprecated {
    pub inherit: u8,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub override1: u8,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub override2: u8,
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
#[deprecated(since = "1.0.0", note = "text")]
pub struct Deprecated2(pub u8,
                       #[stable(feature = "rust1", since = "1.0.0")] pub u8,
                       #[unstable(feature = "unstable_test_feature", issue = "none")] pub u8);
