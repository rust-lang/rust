#![feature(staged_api)]

//@ is "$.index[?(@.name=='non_const_function')].const_stability" null
#[stable(feature = "non_const_function_feature", since = "0.9.0")]
pub fn non_const_function() {}

//@ set stable_const_fn = "$.index[?(@.name=='stable_const_fn')].id"
//@ is "$.index[?(@.name=='stable_const_fn')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='stable_const_fn')].stability.feature" '"stable_const_fn_feature"'
//@ is "$.index[?(@.name=='stable_const_fn')].stability.since" '"1.0.0"'
//@ is "$.index[?(@.name=='stable_const_fn')].const_stability.level" '"stable"'
//@ is "$.index[?(@.name=='stable_const_fn')].const_stability.feature" '"stable_const_fn_const_feature"'
//@ is "$.index[?(@.name=='stable_const_fn')].const_stability.since" '"1.1.0"'
//@ is "$.index[?(@.name=='stable_const_fn')].attrs" []
#[stable(feature = "stable_const_fn_feature", since = "1.0.0")]
#[rustc_const_stable(feature = "stable_const_fn_const_feature", since = "1.1.0")]
pub const fn stable_const_fn() {}

//@ is "$.index[?(@.name=='const_unstable_fn')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='const_unstable_fn')].stability.feature" '"const_unstable_fn_feature"'
//@ is "$.index[?(@.name=='const_unstable_fn')].stability.since" '"2.0.0"'
//@ is "$.index[?(@.name=='const_unstable_fn')].const_stability.level" '"unstable"'
//@ is "$.index[?(@.name=='const_unstable_fn')].const_stability.feature" '"const_unstable_fn_const_feature"'
//@ !has "$.index[?(@.name=='const_unstable_fn')].const_stability.since"
//@ is "$.index[?(@.name=='const_unstable_fn')].attrs" []
#[stable(feature = "const_unstable_fn_feature", since = "2.0.0")]
#[rustc_const_unstable(feature = "const_unstable_fn_const_feature", issue = "none")]
pub const fn const_unstable_fn() {}

// Even when the item itself is unstable, if a separate const-stability attribute is present,
// that's a distinct fact possibly associated with a different feature gate.
// It should therefore be exposed on its own, instead of being collapsed into regular stability.
//@ is "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].stability.feature" '"unstable_fn_with_explicit_const_gate_feature"'
//@ !has "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].stability.since"
//@ is "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].const_stability.level" '"unstable"'
//@ is "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].const_stability.feature" '"explicit_const_gate_on_unstable_fn"'
//@ !has "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].const_stability.since"
//@ is "$.index[?(@.name=='unstable_fn_with_explicit_const_gate')].attrs" []
#[unstable(feature = "unstable_fn_with_explicit_const_gate_feature", issue = "none")]
#[rustc_const_unstable(feature = "explicit_const_gate_on_unstable_fn", issue = "none")]
pub const fn unstable_fn_with_explicit_const_gate() {}

// `lookup_const_stability` synthesizes a const-unstable record for this item from its regular
// instability. Rustdoc JSON filters that out because there is no separate const feature gate.
//@ is "$.index[?(@.name=='unstable_const_fn_without_const_gate')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='unstable_const_fn_without_const_gate')].stability.feature" '"unstable_const_fn_without_const_gate_feature"'
//@ is "$.index[?(@.name=='unstable_const_fn_without_const_gate')].const_stability" null
#[unstable(feature = "unstable_const_fn_without_const_gate_feature", issue = "none")]
pub const fn unstable_const_fn_without_const_gate() {}

// The `Use` item describes the re-export. It doesn't have `const_stability` of its own.
//@ is "$.index[?(@.inner.use.name=='stable_const_fn_reexport')].const_stability" null
//@ is "$.index[?(@.inner.use.name=='stable_const_fn_reexport')].inner.use.id" $stable_const_fn
pub use stable_const_fn as stable_const_fn_reexport;
