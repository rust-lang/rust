#![feature(staged_api)]

// Mirrors standard-library stability on item kinds that are distinct from ordinary functions,
// modules, structs, enums, traits, and impls.

//@ is "$.index[?(@.name=='STABLE_CONST')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='STABLE_CONST')].stability.feature" '"stable_const_feature"'
//@ is "$.index[?(@.name=='STABLE_CONST')].stability.since" '"1.0.0"'
#[stable(feature = "stable_const_feature", since = "1.0.0")]
pub const STABLE_CONST: usize = 0;

//@ is "$.index[?(@.name=='UNSTABLE_STATIC')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='UNSTABLE_STATIC')].stability.feature" '"unstable_static_feature"'
//@ !has "$.index[?(@.name=='UNSTABLE_STATIC')].stability.since"
#[unstable(feature = "unstable_static_feature", issue = "none")]
pub static UNSTABLE_STATIC: usize = 0;

//@ is "$.index[?(@.name=='StableTypeAlias')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='StableTypeAlias')].stability.feature" '"stable_type_alias_feature"'
//@ is "$.index[?(@.name=='StableTypeAlias')].stability.since" '"1.1.0"'
#[stable(feature = "stable_type_alias_feature", since = "1.1.0")]
pub type StableTypeAlias = usize;

//@ is "$.index[?(@.name=='StableUnion')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='StableUnion')].stability.feature" '"stable_union_feature"'
//@ is "$.index[?(@.name=='StableUnion')].stability.since" '"1.3.0"'
#[stable(feature = "stable_union_feature", since = "1.3.0")]
pub union StableUnion {
    storage: usize,
}

//@ is "$.index[?(@.inner.extern_crate.name=='stable_extern_crate_self')].stability.level" '"stable"'
//@ is "$.index[?(@.inner.extern_crate.name=='stable_extern_crate_self')].stability.feature" '"stable_extern_crate_feature"'
//@ is "$.index[?(@.inner.extern_crate.name=='stable_extern_crate_self')].stability.since" '"1.4.0"'
#[stable(feature = "stable_extern_crate_feature", since = "1.4.0")]
pub extern crate self as stable_extern_crate_self;

//@ is "$.index[?(@.name=='unstable_macro_rules')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='unstable_macro_rules')].stability.feature" '"unstable_macro_rules_feature"'
//@ !has "$.index[?(@.name=='unstable_macro_rules')].stability.since"
//@ is "$.index[?(@.name=='unstable_macro_rules')].attrs" '["macro_export"]'
#[unstable(feature = "unstable_macro_rules_feature", issue = "none")]
#[macro_export]
macro_rules! unstable_macro_rules {
    () => {};
}
