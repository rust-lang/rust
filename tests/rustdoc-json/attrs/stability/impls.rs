#![feature(staged_api)]

// Impl blocks and items defined inside impl blocks only. Trait-associated item declarations are
// tested in `associated_items.rs`; any trait items below are scaffolding for impl-item assertions.
// Staged API still requires stability attributes on stable trait items even when the assertions
// below focus on the corresponding impl items.

#[stable(feature = "stable_impl_target", since = "1.0.0")]
pub struct StableImplTarget;

#[stable(feature = "stable_trait_for_impl", since = "1.0.0")]
pub trait StableTraitForImpl {
    #[stable(feature = "stable_trait_output", since = "1.0.0")]
    type StableOutput;

    #[stable(feature = "stable_trait_assoc_const", since = "1.0.0")]
    const STABLE_ASSOC_CONST: usize;

    #[stable(feature = "stable_trait_method", since = "1.0.0")]
    fn stable_trait_method(&self);
}

#[stable(feature = "stable_trait_with_unstable_method_for_impl", since = "1.0.0")]
pub trait StableTraitWithUnstableMethodForImpl {
    #[unstable(feature = "unstable_trait_method_for_stable_impl", issue = "none")]
    fn unstable_trait_method_for_stable_impl(&self);
}

#[unstable(feature = "unstable_trait_for_impl", issue = "none")]
pub trait UnstableTraitForImpl {
    type UnstableOutput;
    const UNSTABLE_ASSOC_CONST: usize;
    fn unstable_trait_method(&self);
}

//@ is "$.index[?(@.docs=='stable inherent impl')].stability.level" '"stable"'
//@ is "$.index[?(@.docs=='stable inherent impl')].stability.feature" '"stable_inherent_impl"'
//@ is "$.index[?(@.docs=='stable inherent impl')].stability.since" '"2.0.0"'
/// stable inherent impl
#[stable(feature = "stable_inherent_impl", since = "2.0.0")]
impl StableImplTarget {
    //@ is "$.index[?(@.name=='stable_inherent_method')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='stable_inherent_method')].stability.feature" '"stable_inherent_method_feature"'
    //@ is "$.index[?(@.name=='stable_inherent_method')].stability.since" '"2.1.0"'
    //@ is "$.index[?(@.name=='stable_inherent_method')].attrs" []
    #[stable(feature = "stable_inherent_method_feature", since = "2.1.0")]
    pub fn stable_inherent_method(&self) {}

    //@ is "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_IMPL')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_IMPL')].stability.feature" '"unstable_assoc_const_in_impl_feature"'
    //@ !has "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_IMPL')].stability.since"
    #[unstable(feature = "unstable_assoc_const_in_impl_feature", issue = "none")]
    pub const UNSTABLE_ASSOC_CONST_IN_IMPL: usize = 2;
}

//@ is "$.index[?(@.docs=='unstable inherent impl')].stability.level" '"unstable"'
//@ is "$.index[?(@.docs=='unstable inherent impl')].stability.feature" '"unstable_inherent_impl"'
//@ !has "$.index[?(@.docs=='unstable inherent impl')].stability.since"
/// unstable inherent impl
#[unstable(feature = "unstable_inherent_impl", issue = "none")]
impl StableImplTarget {
    // The instability of the inherent `impl` block is inherited by the item.
    //@ is "$.index[?(@.name=='method_inside_unstable_inherent_impl')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='method_inside_unstable_inherent_impl')].stability.feature" '"unstable_inherent_impl"'
    //@ !has "$.index[?(@.name=='method_inside_unstable_inherent_impl')].stability.since"
    pub fn method_inside_unstable_inherent_impl(&self) {}
}

// For trait impl items, `stability: null` means the implementation item has no separate
// stability attribute of its own. To decide whether the implemented method/type/const is usable,
// consumers need to combine the trait item stability with the impl block stability.
//
// The stable impl below intentionally does not copy either stable or unstable trait-item
// stability onto the implemented item. The unstable impl case is different: rustc records the
// impl block's instability on contained impl items, so JSON exposes that inherited instability.

//@ is "$.index[?(@.docs=='stable trait impl with unstable trait method')].stability.level" '"stable"'
//@ is "$.index[?(@.docs=='stable trait impl with unstable trait method')].stability.feature" '"stable_trait_impl_with_unstable_trait_method"'
//@ is "$.index[?(@.docs=='stable trait impl with unstable trait method')].stability.since" '"3.0.0"'
/// stable trait impl with unstable trait method
#[stable(feature = "stable_trait_impl_with_unstable_trait_method", since = "3.0.0")]
impl StableTraitWithUnstableMethodForImpl for StableImplTarget {
    // The impl item has no separate stability record; the trait method is unstable.
    //@ is "$.index[?(@.docs=='method for unstable trait item inside stable trait impl')].stability" null
    /// method for unstable trait item inside stable trait impl
    fn unstable_trait_method_for_stable_impl(&self) {}
}

//@ is "$.index[?(@.docs=='stable trait impl')].stability.level" '"stable"'
//@ is "$.index[?(@.docs=='stable trait impl')].stability.feature" '"stable_trait_impl"'
//@ is "$.index[?(@.docs=='stable trait impl')].stability.since" '"3.0.0"'
/// stable trait impl
#[stable(feature = "stable_trait_impl", since = "3.0.0")]
impl StableTraitForImpl for StableImplTarget {
    // These impl items likewise have no separate stability records.
    // Their trait item declarations and the impl block are all stable.

    //@ is "$.index[?(@.docs=='assoc type inside stable trait impl')].stability" null
    /// assoc type inside stable trait impl
    type StableOutput = usize;

    //@ is "$.index[?(@.docs=='assoc const inside stable trait impl')].stability" null
    /// assoc const inside stable trait impl
    const STABLE_ASSOC_CONST: usize = 0;

    //@ is "$.index[?(@.docs=='method inside stable trait impl')].stability" null
    /// method inside stable trait impl
    fn stable_trait_method(&self) {}
}

//@ is "$.index[?(@.docs=='unstable trait impl')].stability.level" '"unstable"'
//@ is "$.index[?(@.docs=='unstable trait impl')].stability.feature" '"unstable_trait_impl"'
//@ !has "$.index[?(@.docs=='unstable trait impl')].stability.since"
/// unstable trait impl
#[unstable(feature = "unstable_trait_impl", issue = "none")]
impl UnstableTraitForImpl for StableImplTarget {
    // These associated items inherit the impl block's instability in rustdoc JSON.

    //@ is "$.index[?(@.docs=='assoc type inside unstable trait impl')].stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='assoc type inside unstable trait impl')].stability.feature" '"unstable_trait_impl"'
    //@ !has "$.index[?(@.docs=='assoc type inside unstable trait impl')].stability.since"
    /// assoc type inside unstable trait impl
    type UnstableOutput = usize;

    //@ is "$.index[?(@.docs=='assoc const inside unstable trait impl')].stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='assoc const inside unstable trait impl')].stability.feature" '"unstable_trait_impl"'
    //@ !has "$.index[?(@.docs=='assoc const inside unstable trait impl')].stability.since"
    /// assoc const inside unstable trait impl
    const UNSTABLE_ASSOC_CONST: usize = 0;

    //@ is "$.index[?(@.docs=='method inside unstable trait impl')].stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='method inside unstable trait impl')].stability.feature" '"unstable_trait_impl"'
    //@ !has "$.index[?(@.docs=='method inside unstable trait impl')].stability.since"
    /// method inside unstable trait impl
    fn unstable_trait_method(&self) {}
}
