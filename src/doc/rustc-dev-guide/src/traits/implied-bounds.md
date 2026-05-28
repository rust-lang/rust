# Implied bounds

We currently add implied region bounds to avoid explicit annotations. e.g.
`fn foo<'a, T>(x: &'a T)` can freely assume that `T: 'a` holds without specifying it.

There are two kinds of implied bounds: explicit and implicit. Explicit implied bounds
get added to the `fn predicates_of` of the relevant item while implicit ones are
handled... well... implicitly.

## explicit implied bounds

The explicit implied bounds are computed in [`fn inferred_outlives_of`]. Only ADTs and
lazy type aliases have explicit implied bounds which are computed via a fixpoint algorithm
in the [`fn inferred_outlives_crate`] query.

We use [`fn insert_required_predicates_to_be_wf`] on all fields of all ADTs in the crate.
This function computes the outlives bounds for each component of the field using a
separate implementation.

For ADTs, trait objects, and associated types the initially required predicates are
computed in [`fn check_explicit_predicates`]. This simply uses `fn explicit_predicates_of`
without elaborating them.

Region predicates are added via [`fn insert_outlives_predicate`]. This function takes
an outlives predicate, decomposes it and adds the components as explicit predicates only
if the outlived region is a region parameter. [It does not add `'static` requirements][nostatic].

 [`fn inferred_outlives_of`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/mod.rs#L20
 [`fn inferred_outlives_crate`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/mod.rs#L83
 [`fn insert_required_predicates_to_be_wf`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/implicit_infer.rs#L89
 [`fn check_explicit_predicates`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/implicit_infer.rs#L238
 [`fn insert_outlives_predicate`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/utils.rs#L15
 [nostatic]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_hir_analysis/src/outlives/utils.rs#L159-L165

## implicit implied bounds

As we are unable to handle implications in binders yet, we cannot simply add the outlives
requirements of impls and functions as explicit predicates.

### using implicit implied bounds as assumptions

These bounds are not added to the `ParamEnv` of the affected item itself. For lexical
region resolution they are added using [`fn OutlivesEnvironment::from_normalized_bounds`].
Similarly, during MIR borrowck we add them using
[`fn UniversalRegionRelationsBuilder::add_implied_bounds`].

[We add implied bounds for the function signature and impl header in MIR borrowck][mir].
Outside of MIR borrowck we add the outlives requirements for the types returned by the
[`fn assumed_wf_types`] query.

The assumed outlives constraints for implicit bounds are computed using the
[`fn implied_outlives_bounds`] query. This directly
[extracts the required outlives bounds from `fn wf::obligations`][boundsfromty].

MIR borrowck adds the outlives constraints for both the normalized and unnormalized types,
lexical region resolution [only uses the unnormalized types][notnorm].

[`fn OutlivesEnvironment::from_normalized_bounds`]: https://github.com/rust-lang/rust/blob/8239a37f9c0951a037cfc51763ea52a20e71e6bd/compiler/rustc_infer/src/infer/outlives/env.rs#L50-L55
[`fn UniversalRegionRelationsBuilder::add_implied_bounds`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_borrowck/src/type_check/free_region_relations.rs#L316
[mir]: https://github.com/rust-lang/rust/blob/91cae1dcdcf1a31bd8a92e4a63793d65cfe289bb/compiler/rustc_borrowck/src/type_check/free_region_relations.rs#L258-L332
[`fn assumed_wf_types`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_ty_utils/src/implied_bounds.rs#L21
[`fn implied_outlives_bounds`]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_traits/src/implied_outlives_bounds.rs#L18C4-L18C27
[boundsfromty]: https://github.com/rust-lang/rust/blob/5b8bc568d28b2e922290c9a966b3231d0ce9398b/compiler/rustc_trait_selection/src/traits/query/type_op/implied_outlives_bounds.rs#L95-L96
[notnorm]: https://github.com/rust-lang/rust/blob/91cae1dcdcf1a31bd8a92e4a63793d65cfe289bb/compiler/rustc_trait_selection/src/traits/engine.rs#L227-L250

### proving implicit implied bounds

As the implicit implied bounds are not included in `fn predicates_of` we have to
separately make sure they actually hold. We generally handle this by checking that
all used types are well formed by emitting `WellFormed` predicates.

We cannot emit `WellFormed` predicates when instantiating impls, as this would result
in - currently often inductive - trait solver cycles. We also do not emit constraints
involving higher ranked regions as we're lacking the implied bounds from their binder.

This results in multiple unsoundnesses:
- by using subtyping: [#25860]
- by using super trait upcasting for a higher ranked trait bound: [#84591]
- by being able to normalize a projection when using an impl while not being able
  to normalize it when checking the impl: [#100051]

[#25860]: https://github.com/rust-lang/rust/issues/25860
[#84591]: https://github.com/rust-lang/rust/issues/84591
[#100051]: https://github.com/rust-lang/rust/issues/100051
