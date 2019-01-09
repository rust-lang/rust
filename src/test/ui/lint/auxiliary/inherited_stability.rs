#![crate_name="inherited_stability"]
#![crate_type = "lib"]
#![unstable(feature = "unstable_test_feature", issue = "0")]
#![feature(staged_api)]

pub fn unstable() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub fn stable() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod stable_mod {
    #[unstable(feature = "unstable_test_feature", issue = "0")]
    pub fn unstable() {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn stable() {}
}

#[unstable(feature = "unstable_test_feature", issue = "0")]
pub mod unstable_mod {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    pub fn deprecated() {}

    pub fn unstable() {}
}

#[stable(feature = "rust1", since = "1.0.0")]
pub trait Stable {
    #[unstable(feature = "unstable_test_feature", issue = "0")]
    fn unstable(&self);

    #[stable(feature = "rust1", since = "1.0.0")]
    fn stable(&self);
}

impl Stable for usize {
    fn unstable(&self) {}
    fn stable(&self) {}
}

pub enum Unstable {
    UnstableVariant,
    #[stable(feature = "rust1", since = "1.0.0")]
    StableVariant
}
