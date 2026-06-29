//! Module items, including the crate root and containing-module stability facts that consumers
//! need when deciding whether a particular path is stable.

#![feature(staged_api)]
#![stable(feature = "stable_crate_feature", since = "1.0.0")]
//@ is "$.index[?(@.name=='modules')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='modules')].stability.feature" '"stable_crate_feature"'
//@ is "$.index[?(@.name=='modules')].stability.since" '"1.0.0"'

pub mod inner_stable_module {
    #![stable(feature = "inner_stable_module_feature", since = "1.1.0")]

    //@ is "$.index[?(@.name=='inner_stable_module')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='inner_stable_module')].stability.feature" '"inner_stable_module_feature"'
    //@ is "$.index[?(@.name=='inner_stable_module')].stability.since" '"1.1.0"'
}

#[unstable(feature = "unstable_module_feature", issue = "none")]
pub mod unstable_module {
    //@ is "$.index[?(@.name=='unstable_module')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_module')].stability.feature" '"unstable_module_feature"'
    //@ !has "$.index[?(@.name=='unstable_module')].stability.since"
}

#[stable(feature = "stable_parent_feature", since = "2.0.0")]
pub mod stable_parent {
    //@ is "$.index[?(@.name=='stable_parent')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='stable_parent')].stability.feature" '"stable_parent_feature"'
    //@ is "$.index[?(@.name=='stable_parent')].stability.since" '"2.0.0"'

    //@ is "$.index[?(@.name=='UnstableChildInStable')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UnstableChildInStable')].stability.feature" '"unstable_child_in_stable"'
    //@ !has "$.index[?(@.name=='UnstableChildInStable')].stability.since"
    #[unstable(feature = "unstable_child_in_stable", issue = "none")]
    pub struct UnstableChildInStable;
}

#[unstable(feature = "unstable_parent_feature", issue = "27")]
pub mod unstable_parent {
    //@ is "$.index[?(@.name=='unstable_parent')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_parent')].stability.feature" '"unstable_parent_feature"'
    //@ !has "$.index[?(@.name=='unstable_parent')].stability.since"

    //@ is "$.index[?(@.name=='StableChildInUnstable')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='StableChildInUnstable')].stability.feature" '"stable_child_in_unstable"'
    //@ is "$.index[?(@.name=='StableChildInUnstable')].stability.since" '"3.0.0"'
    #[stable(feature = "stable_child_in_unstable", since = "3.0.0")]
    pub struct StableChildInUnstable;
}
