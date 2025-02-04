#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![stable(feature = "stable_test_feature", since = "1.2.0")]

#[unstable(feature = "unstable_test_feature", issue = "1")]
pub mod unstable_module {
    #[stable(feature = "stable_test_feature", since = "1.2.0")]
    #[rustc_allowed_through_unstable_modules = "use the new path instead"]
    pub trait OldStableTraitAllowedThoughUnstable {}

    #[stable(feature = "stable_test_feature", since = "1.2.0")]
    pub trait NewStableTraitNotAllowedThroughUnstable {}
}
