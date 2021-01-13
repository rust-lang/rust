#![crate_name="lint_stability"]
#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(associated_type_defaults)]
#![stable(feature = "lint_stability", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated() {}
#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated_text() {}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "99.99.99", reason = "text")]
pub fn deprecated_future() {}

#[unstable(feature = "unstable_test_feature", issue = "none")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated_unstable() {}
#[unstable(feature = "unstable_test_feature", issue = "none")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated_unstable_text() {}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub fn unstable() {}
#[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
pub fn unstable_text() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub fn stable() {}
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stable_text() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct MethodTester;

impl MethodTester {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated(&self) {}
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated_text(&self) {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated_unstable(&self) {}
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated_unstable_text(&self) {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    pub fn method_unstable(&self) {}
    #[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
    pub fn method_unstable_text(&self) {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn method_stable(&self) {}
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn method_stable_text(&self) {}
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated(&self) {}
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated_text(&self) {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated_unstable(&self) {}
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated_unstable_text(&self) {}

    #[unstable(feature = "unstable_test_feature", issue = "none")]
    fn trait_unstable(&self) {}
    #[unstable(feature = "unstable_test_feature", reason = "text", issue = "none")]
    fn trait_unstable_text(&self) {}

    #[stable(feature = "rust1", since = "1.0.0")]
    fn trait_stable(&self) {}
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trait_stable_text(&self) {}
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait TraitWithAssociatedTypes {
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    type TypeUnstable = u8;
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    type TypeDeprecated = u8;
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
impl Trait for MethodTester {}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub trait UnstableTrait { fn dummy(&self) { } }

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub trait DeprecatedTrait {
    #[stable(feature = "stable_test_feature", since = "1.0.0")] fn dummy(&self) { }
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")] pub i: isize
}
#[unstable(feature = "unstable_test_feature", issue = "none")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedUnstableStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")] pub i: isize
}
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct UnstableStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")] pub i: isize
}
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StableStruct {
    #[stable(feature = "stable_test_feature", since = "1.0.0")] pub i: isize
}
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub enum UnstableEnum {}
#[stable(feature = "rust1", since = "1.0.0")]
pub enum StableEnum {}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedUnitStruct;
#[unstable(feature = "unstable_test_feature", issue = "none")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedUnstableUnitStruct;
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct UnstableUnitStruct;
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StableUnitStruct;

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    DeprecatedVariant,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    DeprecatedUnstableVariant,
    #[unstable(feature = "unstable_test_feature", issue = "none")]
    UnstableVariant,

    #[stable(feature = "rust1", since = "1.0.0")]
    StableVariant,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedTupleStruct(#[stable(feature = "rust1", since = "1.0.0")] pub isize);
#[unstable(feature = "unstable_test_feature", issue = "none")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct DeprecatedUnstableTupleStruct(#[stable(feature = "rust1", since = "1.0.0")] pub isize);
#[unstable(feature = "unstable_test_feature", issue = "none")]
pub struct UnstableTupleStruct(#[stable(feature = "rust1", since = "1.0.0")] pub isize);
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StableTupleStruct(#[stable(feature = "rust1", since = "1.0.0")] pub isize);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[macro_export]
macro_rules! macro_test {
    () => (deprecated());
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[macro_export]
macro_rules! macro_test_arg {
    ($func:expr) => ($func);
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[macro_export]
macro_rules! macro_test_arg_nested {
    ($func:ident) => (macro_test_arg!($func()));
}
