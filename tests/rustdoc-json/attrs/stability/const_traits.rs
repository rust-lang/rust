#![feature(staged_api, const_trait_impl)]

#[stable(feature = "const_trait_target_feature", since = "1.0.0")]
pub struct ConstTraitTarget;

#[stable(feature = "const_stable_trait_target_feature", since = "1.0.0")]
pub struct ConstStableTraitTarget;

#[stable(feature = "inherent_const_method_target_feature", since = "1.1.0")]
pub struct InherentConstMethodTarget;

impl InherentConstMethodTarget {
    // This item is plain unstable, not const-unstable specifically.
    // Rustdoc JSON should populate `stability` only, not both it and `const_stability`.
    // Duplicating the info would just add noise to the JSON file.
    //@ is "$.index[?(@.name=='unstable_inherent_const_method_without_const_gate')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_inherent_const_method_without_const_gate')].stability.feature" '"unstable_inherent_const_method_without_const_gate"'
    //@ is "$.index[?(@.name=='unstable_inherent_const_method_without_const_gate')].const_stability" null
    #[unstable(feature = "unstable_inherent_const_method_without_const_gate", issue = "none")]
    pub const fn unstable_inherent_const_method_without_const_gate(&self) {}
}

//@ is "$.index[?(@.name=='ConstTrait')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='ConstTrait')].stability.feature" '"const_trait_feature"'
//@ is "$.index[?(@.name=='ConstTrait')].stability.since" '"2.0.0"'
//@ is "$.index[?(@.name=='ConstTrait')].const_stability.level" '"unstable"'
//@ is "$.index[?(@.name=='ConstTrait')].const_stability.feature" '"const_trait_const_feature"'
//@ !has "$.index[?(@.name=='ConstTrait')].const_stability.since"
//@ is "$.index[?(@.name=='ConstTrait')].attrs" []
#[stable(feature = "const_trait_feature", since = "2.0.0")]
#[rustc_const_unstable(feature = "const_trait_const_feature", issue = "none")]
pub const trait ConstTrait {
    // This item is *not* usable in a const context, because its parent is const-unstable.
    //@ is "$.index[?(@.docs=='assoc type inside const trait')].const_stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='assoc type inside const trait')].const_stability.feature" '"const_trait_const_feature"'
    //@ !has "$.index[?(@.docs=='assoc type inside const trait')].const_stability.since"
    /// assoc type inside const trait
    #[stable(feature = "trait_assoc_type_feature", since = "2.1.0")]
    type TraitAssocType;

    // This item is also *not* usable in a const context, because its parent is const-unstable.
    //@ is "$.index[?(@.docs=='method inside const trait')].const_stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='method inside const trait')].const_stability.feature" '"const_trait_const_feature"'
    //@ !has "$.index[?(@.docs=='method inside const trait')].const_stability.since"
    /// method inside const trait
    #[stable(feature = "trait_method_feature", since = "2.2.0")]
    fn trait_method(&self);

    // An associated const can't be const-unstable.
    // It is usable in a const context *regardless* of the fact its parent is const-unstable.
    // See the UI test in `tests/ui/traits/const-traits/associated-const-stability.rs` for proof.
    //@ is "$.index[?(@.docs=='assoc const inside const trait')].const_stability" null
    /// assoc const inside const trait
    #[stable(feature = "trait_assoc_const_feature", since = "2.3.0")]
    const TRAIT_ASSOC_CONST: usize;
}

//@ is "$.index[?(@.name=='ConstTraitWithUnstableMethod')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='ConstTraitWithUnstableMethod')].const_stability.level" '"unstable"'
//@ is "$.index[?(@.name=='ConstTraitWithUnstableMethod')].const_stability.feature" '"const_trait_with_unstable_method_const_feature"'
#[stable(feature = "const_trait_with_unstable_method_feature", since = "2.4.0")]
#[rustc_const_unstable(feature = "const_trait_with_unstable_method_const_feature", issue = "none")]
pub const trait ConstTraitWithUnstableMethod {
    // The method has its own regular instability, but no separate const-stability attribute.
    // Even if it became stable, the parent's const-instability would still dominate and
    // keep it const-unstable. Rustdoc JSON should expose the inherited const-instability.
    //@ is "$.index[?(@.name=='unstable_method_without_const_gate')].stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_method_without_const_gate')].stability.feature" '"unstable_method_without_const_gate"'
    //@ is "$.index[?(@.name=='unstable_method_without_const_gate')].const_stability.level" '"unstable"'
    //@ is "$.index[?(@.name=='unstable_method_without_const_gate')].const_stability.feature" '"const_trait_with_unstable_method_const_feature"'
    #[unstable(feature = "unstable_method_without_const_gate", issue = "none")]
    fn unstable_method_without_const_gate(&self);
}

//@ is "$.index[?(@.docs=='const trait impl')].stability.level" '"stable"'
//@ is "$.index[?(@.docs=='const trait impl')].stability.feature" '"const_trait_impl_feature"'
//@ is "$.index[?(@.docs=='const trait impl')].stability.since" '"3.0.0"'
//@ is "$.index[?(@.docs=='const trait impl')].const_stability.level" '"unstable"'
//@ is "$.index[?(@.docs=='const trait impl')].const_stability.feature" '"const_trait_impl_const_feature"'
//@ !has "$.index[?(@.docs=='const trait impl')].const_stability.since"
/// const trait impl
#[stable(feature = "const_trait_impl_feature", since = "3.0.0")]
#[rustc_const_unstable(feature = "const_trait_impl_const_feature", issue = "none")]
const impl ConstTrait for ConstTraitTarget {
    // This item is *not* usable in a const context, because its parent is const-unstable.
    //@ is "$.index[?(@.docs=='assoc type inside const trait impl')].const_stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='assoc type inside const trait impl')].const_stability.feature" '"const_trait_impl_const_feature"'
    //@ !has "$.index[?(@.docs=='assoc type inside const trait impl')].const_stability.since"
    /// assoc type inside const trait impl
    type TraitAssocType = usize;

    // This item is also *not* usable in a const context, because its parent is const-unstable.
    //@ is "$.index[?(@.docs=='method inside const trait impl')].const_stability.level" '"unstable"'
    //@ is "$.index[?(@.docs=='method inside const trait impl')].const_stability.feature" '"const_trait_impl_const_feature"'
    //@ !has "$.index[?(@.docs=='method inside const trait impl')].const_stability.since"
    /// method inside const trait impl
    fn trait_method(&self) {}

    // An associated const can't be const-unstable.
    // It is usable in a const context *regardless* of the fact its parent is const-unstable.
    // See the UI test in `tests/ui/traits/const-traits/associated-const-stability.rs` for proof.
    //@ is "$.index[?(@.docs=='assoc const inside const trait impl')].const_stability" null
    /// assoc const inside const trait impl
    const TRAIT_ASSOC_CONST: usize = 0;
}

// The `const trait` is const-stable. We indicate that *once* on the trait, but since
// there are no special cases here (unlike in the const-unstable case with associated consts)
// we do not duplicate the const-stability to all the associated items so as not to bloat the JSON.
//@ is "$.index[?(@.name=='ConstStableTrait')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='ConstStableTrait')].stability.feature" '"const_stable_trait_feature"'
//@ is "$.index[?(@.name=='ConstStableTrait')].stability.since" '"4.0.0"'
//@ is "$.index[?(@.name=='ConstStableTrait')].const_stability.level" '"stable"'
//@ is "$.index[?(@.name=='ConstStableTrait')].const_stability.since" '"4.0.0"'
//@ is "$.index[?(@.name=='ConstStableTrait')].const_stability.feature" '"const_stable_trait_const_feature"'
#[stable(feature = "const_stable_trait_feature", since = "4.0.0")]
#[rustc_const_stable(feature = "const_stable_trait_const_feature", since = "4.0.0")]
pub const trait ConstStableTrait {
    //@ is "$.index[?(@.docs=='assoc type inside const-stable trait')].const_stability" null
    /// assoc type inside const-stable trait
    #[stable(feature = "stable_trait_assoc_type_feature", since = "4.1.0")]
    type StableTraitAssocType;

    //@ is "$.index[?(@.docs=='method inside const-stable trait')].const_stability" null
    /// method inside const-stable trait
    #[stable(feature = "stable_trait_method_feature", since = "4.2.0")]
    fn stable_trait_method(&self);

    //@ is "$.index[?(@.docs=='assoc const inside const-stable trait')].const_stability" null
    /// assoc const inside const-stable trait
    #[stable(feature = "stable_trait_assoc_const_feature", since = "4.3.0")]
    const STABLE_TRAIT_ASSOC_CONST: usize;
}

// The `const impl` is const-stable. We indicate that *once* on the impl itself, but since
// there are no special cases here (unlike in the const-unstable case with associated consts)
// we do not duplicate the const-stability to all the associated items so as not to bloat the JSON.
//@ is "$.index[?(@.docs=='const-stable trait impl')].stability.level" '"stable"'
//@ is "$.index[?(@.docs=='const-stable trait impl')].stability.feature" '"const_stable_trait_impl_feature"'
//@ is "$.index[?(@.docs=='const-stable trait impl')].stability.since" '"5.0.0"'
//@ is "$.index[?(@.docs=='const-stable trait impl')].const_stability.level" '"stable"'
//@ is "$.index[?(@.docs=='const-stable trait impl')].const_stability.since" '"5.0.0"'
//@ is "$.index[?(@.docs=='const-stable trait impl')].const_stability.feature" '"const_stable_trait_impl_const_feature"'
/// const-stable trait impl
#[stable(feature = "const_stable_trait_impl_feature", since = "5.0.0")]
#[rustc_const_stable(feature = "const_stable_trait_impl_const_feature", since = "5.0.0")]
const impl ConstStableTrait for ConstStableTraitTarget {
    //@ is "$.index[?(@.docs=='assoc type inside const-stable trait impl')].const_stability" null
    /// assoc type inside const-stable trait impl
    type StableTraitAssocType = usize;

    //@ is "$.index[?(@.docs=='method inside const-stable trait impl')].const_stability" null
    /// method inside const-stable trait impl
    fn stable_trait_method(&self) {}

    //@ is "$.index[?(@.docs=='assoc const inside const-stable trait impl')].const_stability" null
    /// assoc const inside const-stable trait impl
    const STABLE_TRAIT_ASSOC_CONST: usize = 1;
}
