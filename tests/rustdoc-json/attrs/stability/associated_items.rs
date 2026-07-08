#![feature(staged_api)]

// Trait-associated item declarations only. Items defined inside impl blocks are tested in
// `impls.rs` because they have different parent-item behavior.

//@ is "$.index[?(@.name=='StableTraitWithAssociatedItems')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='StableTraitWithAssociatedItems')].stability.feature" '"stable_trait_with_associated_items"'
//@ is "$.index[?(@.name=='StableTraitWithAssociatedItems')].stability.since" '"1.0.0"'
#[stable(feature = "stable_trait_with_associated_items", since = "1.0.0")]
pub trait StableTraitWithAssociatedItems {
    // Stable trait-associated items need their own stability attributes in staged API crates.
    //@ is "$.index[?(@.name=='StableAssocType')].stability.level" '"stable"'
    //@ is "$.index[?(@.name=='StableAssocType')].stability.feature" '"stable_assoc_type_feature"'
    //@ is "$.index[?(@.name=='StableAssocType')].stability.since" '"1.1.0"'
    //@ is "$.index[?(@.name=='StableAssocType')].attrs" []
    #[stable(feature = "stable_assoc_type_feature", since = "1.1.0")]
    type StableAssocType;

    //@ is "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_STABLE_TRAIT')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_STABLE_TRAIT')].stability.feature" '"unstable_assoc_const_in_stable_trait"'
    //@ !has "$.index[?(@.name=='UNSTABLE_ASSOC_CONST_IN_STABLE_TRAIT')].stability.since"
    #[unstable(feature = "unstable_assoc_const_in_stable_trait", issue = "none")]
    const UNSTABLE_ASSOC_CONST_IN_STABLE_TRAIT: usize = 0;

    //@ is "$.index[?(@.name=='unstable_provided_method')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_provided_method')].stability.feature" '"unstable_provided_method_feature"'
    //@ !has "$.index[?(@.name=='unstable_provided_method')].stability.since"
    #[unstable(feature = "unstable_provided_method_feature", issue = "none")]
    fn unstable_provided_method(&self) {}
}

//@ is "$.index[?(@.name=='UnstableTraitWithUnannotatedAssociatedItems')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='UnstableTraitWithUnannotatedAssociatedItems')].stability.feature" '"unstable_trait_with_unannotated_associated_items"'
//@ !has "$.index[?(@.name=='UnstableTraitWithUnannotatedAssociatedItems')].stability.since"
#[unstable(feature = "unstable_trait_with_unannotated_associated_items", issue = "none")]
pub trait UnstableTraitWithUnannotatedAssociatedItems {
    // Unannotated associated items in unstable traits inherit the trait's unstable stability.
    //@ is "$.index[?(@.name=='UnannotatedAssocTypeInUnstableTrait')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UnannotatedAssocTypeInUnstableTrait')].stability.feature" '"unstable_trait_with_unannotated_associated_items"'
    //@ !has "$.index[?(@.name=='UnannotatedAssocTypeInUnstableTrait')].stability.since"
    type UnannotatedAssocTypeInUnstableTrait;

    //@ is "$.index[?(@.name=='UNANNOTATED_ASSOC_CONST_IN_UNSTABLE_TRAIT')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UNANNOTATED_ASSOC_CONST_IN_UNSTABLE_TRAIT')].stability.feature" '"unstable_trait_with_unannotated_associated_items"'
    //@ !has "$.index[?(@.name=='UNANNOTATED_ASSOC_CONST_IN_UNSTABLE_TRAIT')].stability.since"
    const UNANNOTATED_ASSOC_CONST_IN_UNSTABLE_TRAIT: usize;

    //@ is "$.index[?(@.name=='unannotated_required_method_in_unstable_trait')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unannotated_required_method_in_unstable_trait')].stability.feature" '"unstable_trait_with_unannotated_associated_items"'
    //@ !has "$.index[?(@.name=='unannotated_required_method_in_unstable_trait')].stability.since"
    fn unannotated_required_method_in_unstable_trait(&self);
}

//@ is "$.index[?(@.name=='UnstableTraitWithExplicitAssociatedItem')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='UnstableTraitWithExplicitAssociatedItem')].stability.feature" '"unstable_trait_with_explicit_associated_item"'
//@ !has "$.index[?(@.name=='UnstableTraitWithExplicitAssociatedItem')].stability.since"
#[unstable(feature = "unstable_trait_with_explicit_associated_item", issue = "none")]
pub trait UnstableTraitWithExplicitAssociatedItem {
    // It's possble to override the parent's instability with another `#[unstable]` attribute,
    // for example to specify a different feature gate for that item.
    //@ is "$.index[?(@.name=='UnstableAssocTypeInUnstableTrait')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='UnstableAssocTypeInUnstableTrait')].stability.feature" '"unstable_assoc_type_in_unstable_trait"'
    //@ !has "$.index[?(@.name=='UnstableAssocTypeInUnstableTrait')].stability.since"
    #[unstable(feature = "unstable_assoc_type_in_unstable_trait", issue = "none")]
    type UnstableAssocTypeInUnstableTrait;
}
