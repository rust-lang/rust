#![feature(staged_api)]

// The stability of an item is a function of both the item and the path by which it is used.
// It's possible for a stable item to be available under both a stable and an unstable path,
// depending on re-exports and module stability. This file tests such edge cases.
//
// To determine if a path is stable, users of rustdoc JSON should walk all path components
// (including `Use` items' `inner.use.id` value) and check for (in)stability.
// The path is unstable if any traversed component (modules, re-exports, or the final item)
// is marked unstable.

#[unstable(feature = "unstable_source_mod", issue = "none")]
pub mod unstable_source_mod {
    //@ set stable_in_unstable = "$.index[?(@.name=='StableInUnstable')].id"
    //@ is "$.index[?(@.name=='StableInUnstable')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='StableInUnstable')].stability.feature" '"stable_in_unstable"'
    //@ is "$.index[?(@.name=='StableInUnstable')].stability.since" '"1.0.0"'
    #[stable(feature = "stable_in_unstable", since = "1.0.0")]
    pub struct StableInUnstable;

    //@ set second_stable_in_unstable = "$.index[?(@.name=='SecondStableInUnstable')].id"
    #[stable(feature = "second_stable_in_unstable", since = "1.1.0")]
    pub struct SecondStableInUnstable;

    #[stable(feature = "glob_stable_in_unstable", since = "1.2.0")]
    pub struct GlobStableInUnstable;
}

//@ is "$.index[?(@.inner.use.name=='ReexportedStableInUnstable')].inner.use.id" $stable_in_unstable
//@ is "$.index[?(@.inner.use.name=='ReexportedStableInUnstable')].stability.level" '"stable"'
//@ is "$.index[?(@.inner.use.name=='ReexportedStableInUnstable')].stability.feature" '"stable_reexport"'
//@ is "$.index[?(@.inner.use.name=='ReexportedStableInUnstable')].stability.since" '"3.0.0"'
//@ is "$.index[?(@.inner.use.name=='ReexportedStableInUnstable')].attrs" []
#[stable(feature = "stable_reexport", since = "3.0.0")]
pub use crate::unstable_source_mod::StableInUnstable as ReexportedStableInUnstable;
//@ is "$.index[?(@.inner.use.source=='crate::unstable_source_mod')].inner.use.is_glob" true
//@ is "$.index[?(@.inner.use.source=='crate::unstable_source_mod')].stability.level" '"stable"'
//@ is "$.index[?(@.inner.use.source=='crate::unstable_source_mod')].stability.feature" '"stable_glob_reexport"'
//@ is "$.index[?(@.inner.use.source=='crate::unstable_source_mod')].stability.since" '"4.0.0"'
#[stable(feature = "stable_glob_reexport", since = "4.0.0")]
pub use crate::unstable_source_mod::*;
//@ is "$.index[?(@.inner.use.name=='GroupedStableReexport')].inner.use.id" $stable_in_unstable
//@ is "$.index[?(@.inner.use.name=='GroupedStableReexport')].stability.level" '"stable"'
//@ is "$.index[?(@.inner.use.name=='GroupedStableReexport')].stability.feature" '"stable_grouped_reexport"'
//@ is "$.index[?(@.inner.use.name=='GroupedStableReexport')].stability.since" '"3.5.0"'
//@ is "$.index[?(@.inner.use.name=='SecondGroupedStableReexport')].inner.use.id" $second_stable_in_unstable
//@ is "$.index[?(@.inner.use.name=='SecondGroupedStableReexport')].stability.level" '"stable"'
//@ is "$.index[?(@.inner.use.name=='SecondGroupedStableReexport')].stability.feature" '"stable_grouped_reexport"'
//@ is "$.index[?(@.inner.use.name=='SecondGroupedStableReexport')].stability.since" '"3.5.0"'
#[stable(feature = "stable_grouped_reexport", since = "3.5.0")]
pub use crate::unstable_source_mod::{
    SecondStableInUnstable as SecondGroupedStableReexport,
    StableInUnstable as GroupedStableReexport,
};

#[unstable(feature = "unstable_reexport_source_mod", issue = "none")]
pub mod unstable_reexport_source_mod {
    //@ set stable_for_unstable_reexport = "$.index[?(@.name=='StableForUnstableReexport')].id"
    #[stable(feature = "stable_for_unstable_reexport", since = "6.0.0")]
    pub struct StableForUnstableReexport;

    //@ set grouped_stable_for_unstable_reexport = "$.index[?(@.name=='GroupedStableForUnstableReexport')].id"
    #[stable(feature = "grouped_stable_for_unstable_reexport", since = "6.1.0")]
    pub struct GroupedStableForUnstableReexport;

    //@ set second_grouped_stable_for_unstable_reexport = "$.index[?(@.name=='SecondGroupedStableForUnstableReexport')].id"
    #[stable(feature = "second_grouped_stable_for_unstable_reexport", since = "6.2.0")]
    pub struct SecondGroupedStableForUnstableReexport;

    #[stable(feature = "glob_stable_for_unstable_reexport", since = "6.3.0")]
    pub struct GlobStableForUnstableReexport;
}

//@ is "$.index[?(@.inner.use.name=='UnstableReexportedStable')].inner.use.id" $stable_for_unstable_reexport
//@ is "$.index[?(@.inner.use.name=='UnstableReexportedStable')].stability.level" '"unstable"'
//@ is "$.index[?(@.inner.use.name=='UnstableReexportedStable')].stability.feature" '"unstable_reexport"'
//@ !has "$.index[?(@.inner.use.name=='UnstableReexportedStable')].stability.since"
#[unstable(feature = "unstable_reexport", issue = "none")]
pub use crate::unstable_reexport_source_mod::StableForUnstableReexport as UnstableReexportedStable;
//@ is "$.index[?(@.inner.use.source=='crate::unstable_reexport_source_mod')].inner.use.is_glob" true
//@ is "$.index[?(@.inner.use.source=='crate::unstable_reexport_source_mod')].stability.level" '"unstable"'
//@ is "$.index[?(@.inner.use.source=='crate::unstable_reexport_source_mod')].stability.feature" '"unstable_glob_reexport"'
//@ !has "$.index[?(@.inner.use.source=='crate::unstable_reexport_source_mod')].stability.since"
#[unstable(feature = "unstable_glob_reexport", issue = "none")]
pub use crate::unstable_reexport_source_mod::*;
//@ is "$.index[?(@.inner.use.name=='GroupedUnstableReexport')].inner.use.id" $grouped_stable_for_unstable_reexport
//@ is "$.index[?(@.inner.use.name=='GroupedUnstableReexport')].stability.level" '"unstable"'
//@ is "$.index[?(@.inner.use.name=='GroupedUnstableReexport')].stability.feature" '"unstable_grouped_reexport"'
//@ !has "$.index[?(@.inner.use.name=='GroupedUnstableReexport')].stability.since"
//@ is "$.index[?(@.inner.use.name=='SecondGroupedUnstableReexport')].inner.use.id" $second_grouped_stable_for_unstable_reexport
//@ is "$.index[?(@.inner.use.name=='SecondGroupedUnstableReexport')].stability.level" '"unstable"'
//@ is "$.index[?(@.inner.use.name=='SecondGroupedUnstableReexport')].stability.feature" '"unstable_grouped_reexport"'
//@ !has "$.index[?(@.inner.use.name=='SecondGroupedUnstableReexport')].stability.since"
#[unstable(feature = "unstable_grouped_reexport", issue = "none")]
pub use crate::unstable_reexport_source_mod::{
    GroupedStableForUnstableReexport as GroupedUnstableReexport,
    SecondGroupedStableForUnstableReexport as SecondGroupedUnstableReexport,
};
