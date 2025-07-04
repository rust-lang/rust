use core::ops::ControlFlow;
use std::borrow::Cow;
use std::iter;

use hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{Applicability, ErrorGuaranteed, MultiSpan, pluralize, struct_span_code_err};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{self as hir, AmbigArg, GenericParamKind, ImplItemKind, intravisit};
use rustc_infer::infer::{self, BoundRegionConversionTime, InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::util;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{
    self, BottomUpFolder, GenericArgs, GenericParamDefKind, Ty, TyCtxt, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitable, TypeVisitableExt, TypeVisitor, TypingMode, Upcast,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{DUMMY_SP, Span};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::{
    self, FulfillmentError, ObligationCause, ObligationCauseCode, ObligationCtxt,
};
use tracing::{debug, instrument};

use super::potentially_plural_count;
use crate::errors::{LifetimesOrBoundsMismatchOnTrait, MethodShouldReturnFuture};

pub(super) mod refine;

/// Call the query `tcx.compare_impl_item()` directly instead.
pub(super) fn compare_impl_item(
    tcx: TyCtxt<'_>,
    impl_item_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let impl_item = tcx.associated_item(impl_item_def_id);
    let trait_item = tcx.associated_item(impl_item.trait_item_def_id.unwrap());
    let impl_trait_ref =
        tcx.impl_trait_ref(impl_item.container_id(tcx)).unwrap().instantiate_identity();
    debug!(?impl_trait_ref);

    match impl_item.kind {
        ty::AssocKind::Fn { .. } => compare_impl_method(tcx, impl_item, trait_item, impl_trait_ref),
        ty::AssocKind::Type { .. } => compare_impl_ty(tcx, impl_item, trait_item, impl_trait_ref),
        ty::AssocKind::Const { .. } => {
            compare_impl_const(tcx, impl_item, trait_item, impl_trait_ref)
        }
    }
}

/// Checks that a method from an impl conforms to the signature of
/// the same method as declared in the trait.
///
/// # Parameters
///
/// - `impl_m`: type of the method we are checking
/// - `trait_m`: the method in the trait
/// - `impl_trait_ref`: the TraitRef corresponding to the trait implementation
#[instrument(level = "debug", skip(tcx))]
fn compare_impl_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    check_method_is_structurally_compatible(tcx, impl_m, trait_m, impl_trait_ref, false)?;
    compare_method_predicate_entailment(tcx, impl_m, trait_m, impl_trait_ref)?;
    Ok(())
}

/// Checks a bunch of different properties of the impl/trait methods for
/// compatibility, such as asyncness, number of argument, self receiver kind,
/// and number of early- and late-bound generics.
fn check_method_is_structurally_compatible<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    compare_self_type(tcx, impl_m, trait_m, impl_trait_ref, delay)?;
    compare_number_of_generics(tcx, impl_m, trait_m, delay)?;
    compare_generic_param_kinds(tcx, impl_m, trait_m, delay)?;
    compare_number_of_method_arguments(tcx, impl_m, trait_m, delay)?;
    compare_synthetic_generics(tcx, impl_m, trait_m, delay)?;
    check_region_bounds_on_impl_item(tcx, impl_m, trait_m, delay)?;
    Ok(())
}

/// This function is best explained by example. Consider a trait with its implementation:
///
/// ```rust
/// trait Trait<'t, T> {
///     // `trait_m`
///     fn method<'a, M>(t: &'t T, m: &'a M) -> Self;
/// }
///
/// struct Foo;
///
/// impl<'i, 'j, U> Trait<'j, &'i U> for Foo {
///     // `impl_m`
///     fn method<'b, N>(t: &'j &'i U, m: &'b N) -> Foo { Foo }
/// }
/// ```
///
/// We wish to decide if those two method types are compatible.
/// For this we have to show that, assuming the bounds of the impl hold, the
/// bounds of `trait_m` imply the bounds of `impl_m`.
///
/// We start out with `trait_to_impl_args`, that maps the trait
/// type parameters to impl type parameters. This is taken from the
/// impl trait reference:
///
/// ```rust,ignore (pseudo-Rust)
/// trait_to_impl_args = {'t => 'j, T => &'i U, Self => Foo}
/// ```
///
/// We create a mapping `dummy_args` that maps from the impl type
/// parameters to fresh types and regions. For type parameters,
/// this is the identity transform, but we could as well use any
/// placeholder types. For regions, we convert from bound to free
/// regions (Note: but only early-bound regions, i.e., those
/// declared on the impl or used in type parameter bounds).
///
/// ```rust,ignore (pseudo-Rust)
/// impl_to_placeholder_args = {'i => 'i0, U => U0, N => N0 }
/// ```
///
/// Now we can apply `placeholder_args` to the type of the impl method
/// to yield a new function type in terms of our fresh, placeholder
/// types:
///
/// ```rust,ignore (pseudo-Rust)
/// <'b> fn(t: &'i0 U0, m: &'b N0) -> Foo
/// ```
///
/// We now want to extract and instantiate the type of the *trait*
/// method and compare it. To do so, we must create a compound
/// instantiation by combining `trait_to_impl_args` and
/// `impl_to_placeholder_args`, and also adding a mapping for the method
/// type parameters. We extend the mapping to also include
/// the method parameters.
///
/// ```rust,ignore (pseudo-Rust)
/// trait_to_placeholder_args = { T => &'i0 U0, Self => Foo, M => N0 }
/// ```
///
/// Applying this to the trait method type yields:
///
/// ```rust,ignore (pseudo-Rust)
/// <'a> fn(t: &'i0 U0, m: &'a N0) -> Foo
/// ```
///
/// This type is also the same but the name of the bound region (`'a`
/// vs `'b`). However, the normal subtyping rules on fn types handle
/// this kind of equivalency just fine.
///
/// We now use these generic parameters to ensure that all declared bounds
/// are satisfied by the implementation's method.
///
/// We do this by creating a parameter environment which contains a
/// generic parameter corresponding to `impl_to_placeholder_args`. We then build
/// `trait_to_placeholder_args` and use it to convert the predicates contained
/// in the `trait_m` generics to the placeholder form.
///
/// Finally we register each of these predicates as an obligation and check that
/// they hold.
#[instrument(level = "debug", skip(tcx, impl_trait_ref))]
fn compare_method_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    // This node-id should be used for the `body_id` field on each
    // `ObligationCause` (and the `FnCtxt`).
    //
    // FIXME(@lcnr): remove that after removing `cause.body_id` from
    // obligations.
    let impl_m_def_id = impl_m.def_id.expect_local();
    let impl_m_span = tcx.def_span(impl_m_def_id);
    let cause = ObligationCause::new(
        impl_m_span,
        impl_m_def_id,
        ObligationCauseCode::CompareImplItem {
            impl_item_def_id: impl_m_def_id,
            trait_item_def_id: trait_m.def_id,
            kind: impl_m.kind,
        },
    );

    // Create mapping from trait method to impl method.
    let impl_def_id = impl_m.container_id(tcx);
    let trait_to_impl_args = GenericArgs::identity_for_item(tcx, impl_m.def_id).rebase_onto(
        tcx,
        impl_m.container_id(tcx),
        impl_trait_ref.args,
    );
    debug!(?trait_to_impl_args);

    let impl_m_predicates = tcx.predicates_of(impl_m.def_id);
    let trait_m_predicates = tcx.predicates_of(trait_m.def_id);

    // This is the only tricky bit of the new way we check implementation methods
    // We need to build a set of predicates where only the method-level bounds
    // are from the trait and we assume all other bounds from the implementation
    // to be previously satisfied.
    //
    // We then register the obligations from the impl_m and check to see
    // if all constraints hold.
    let impl_predicates = tcx.predicates_of(impl_m_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate_identity(tcx).predicates;
    hybrid_preds.extend(
        trait_m_predicates.instantiate_own(tcx, trait_to_impl_args).map(|(predicate, _)| predicate),
    );

    let is_conditionally_const = tcx.is_conditionally_const(impl_def_id);
    if is_conditionally_const {
        // Augment the hybrid param-env with the const conditions
        // of the impl header and the trait method.
        hybrid_preds.extend(
            tcx.const_conditions(impl_def_id)
                .instantiate_identity(tcx)
                .into_iter()
                .chain(
                    tcx.const_conditions(trait_m.def_id).instantiate_own(tcx, trait_to_impl_args),
                )
                .map(|(trait_ref, _)| {
                    trait_ref.to_host_effect_clause(tcx, ty::BoundConstness::Maybe)
                }),
        );
    }

    let normalize_cause = traits::ObligationCause::misc(impl_m_span, impl_m_def_id);
    let param_env = ty::ParamEnv::new(tcx.mk_clauses(&hybrid_preds));
    let param_env = traits::normalize_param_env_or_error(tcx, param_env, normalize_cause);
    debug!(caller_bounds=?param_env.caller_bounds());

    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);

    // Create obligations for each predicate declared by the impl
    // definition in the context of the hybrid param-env. This makes
    // sure that the impl's method's where clauses are not more
    // restrictive than the trait's method (and the impl itself).
    let impl_m_own_bounds = impl_m_predicates.instantiate_own_identity();
    for (predicate, span) in impl_m_own_bounds {
        let normalize_cause = traits::ObligationCause::misc(span, impl_m_def_id);
        let predicate = ocx.normalize(&normalize_cause, param_env, predicate);

        let cause = ObligationCause::new(
            span,
            impl_m_def_id,
            ObligationCauseCode::CompareImplItem {
                impl_item_def_id: impl_m_def_id,
                trait_item_def_id: trait_m.def_id,
                kind: impl_m.kind,
            },
        );
        ocx.register_obligation(traits::Obligation::new(tcx, cause, param_env, predicate));
    }

    // If we're within a const implementation, we need to make sure that the method
    // does not assume stronger `[const]` bounds than the trait definition.
    //
    // This registers the `[const]` bounds of the impl method, which we will prove
    // using the hybrid param-env that we earlier augmented with the const conditions
    // from the impl header and trait method declaration.
    if is_conditionally_const {
        for (const_condition, span) in
            tcx.const_conditions(impl_m.def_id).instantiate_own_identity()
        {
            let normalize_cause = traits::ObligationCause::misc(span, impl_m_def_id);
            let const_condition = ocx.normalize(&normalize_cause, param_env, const_condition);

            let cause = ObligationCause::new(
                span,
                impl_m_def_id,
                ObligationCauseCode::CompareImplItem {
                    impl_item_def_id: impl_m_def_id,
                    trait_item_def_id: trait_m.def_id,
                    kind: impl_m.kind,
                },
            );
            ocx.register_obligation(traits::Obligation::new(
                tcx,
                cause,
                param_env,
                const_condition.to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
            ));
        }
    }

    // We now need to check that the signature of the impl method is
    // compatible with that of the trait method. We do this by
    // checking that `impl_fty <: trait_fty`.
    //
    // FIXME. Unfortunately, this doesn't quite work right now because
    // associated type normalization is not integrated into subtype
    // checks. For the comparison to be valid, we need to
    // normalize the associated types in the impl/trait methods
    // first. However, because function types bind regions, just
    // calling `FnCtxt::normalize` would have no effect on
    // any associated types appearing in the fn arguments or return
    // type.

    let mut wf_tys = FxIndexSet::default();

    let unnormalized_impl_sig = infcx.instantiate_binder_with_fresh_vars(
        impl_m_span,
        BoundRegionConversionTime::HigherRankedType,
        tcx.fn_sig(impl_m.def_id).instantiate_identity(),
    );

    let norm_cause = ObligationCause::misc(impl_m_span, impl_m_def_id);
    let impl_sig = ocx.normalize(&norm_cause, param_env, unnormalized_impl_sig);
    debug!(?impl_sig);

    let trait_sig = tcx.fn_sig(trait_m.def_id).instantiate(tcx, trait_to_impl_args);
    let trait_sig = tcx.liberate_late_bound_regions(impl_m.def_id, trait_sig);

    // Next, add all inputs and output as well-formed tys. Importantly,
    // we have to do this before normalization, since the normalized ty may
    // not contain the input parameters. See issue #87748.
    wf_tys.extend(trait_sig.inputs_and_output.iter());
    let trait_sig = ocx.normalize(&norm_cause, param_env, trait_sig);
    // We also have to add the normalized trait signature
    // as we don't normalize during implied bounds computation.
    wf_tys.extend(trait_sig.inputs_and_output.iter());
    debug!(?trait_sig);

    // FIXME: We'd want to keep more accurate spans than "the method signature" when
    // processing the comparison between the trait and impl fn, but we sadly lose them
    // and point at the whole signature when a trait bound or specific input or output
    // type would be more appropriate. In other places we have a `Vec<Span>`
    // corresponding to their `Vec<Predicate>`, but we don't have that here.
    // Fixing this would improve the output of test `issue-83765.rs`.
    let result = ocx.sup(&cause, param_env, trait_sig, impl_sig);

    if let Err(terr) = result {
        debug!(?impl_sig, ?trait_sig, ?terr, "sub_types failed");

        let emitted = report_trait_method_mismatch(
            infcx,
            cause,
            param_env,
            terr,
            (trait_m, trait_sig),
            (impl_m, impl_sig),
            impl_trait_ref,
        );
        return Err(emitted);
    }

    if !(impl_sig, trait_sig).references_error() {
        ocx.register_obligation(traits::Obligation::new(
            infcx.tcx,
            cause,
            param_env,
            ty::ClauseKind::WellFormed(
                Ty::new_fn_ptr(tcx, ty::Binder::dummy(unnormalized_impl_sig)).into(),
            ),
        ));
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let reported = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(reported);
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters.
    let errors = infcx.resolve_regions(impl_m_def_id, param_env, wf_tys);
    if !errors.is_empty() {
        return Err(infcx
            .tainted_by_errors()
            .unwrap_or_else(|| infcx.err_ctxt().report_region_errors(impl_m_def_id, &errors)));
    }

    Ok(())
}

struct RemapLateParam<'tcx> {
    tcx: TyCtxt<'tcx>,
    mapping: FxIndexMap<ty::LateParamRegionKind, ty::LateParamRegionKind>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for RemapLateParam<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReLateParam(fr) = r.kind() {
            ty::Region::new_late_param(
                self.tcx,
                fr.scope,
                self.mapping.get(&fr.kind).copied().unwrap_or(fr.kind),
            )
        } else {
            r
        }
    }
}

/// Given a method def-id in an impl, compare the method signature of the impl
/// against the trait that it's implementing. In doing so, infer the hidden types
/// that this method's signature provides to satisfy each return-position `impl Trait`
/// in the trait signature.
///
/// The method is also responsible for making sure that the hidden types for each
/// RPITIT actually satisfy the bounds of the `impl Trait`, i.e. that if we infer
/// `impl Trait = Foo`, that `Foo: Trait` holds.
///
/// For example, given the sample code:
///
/// ```
/// use std::ops::Deref;
///
/// trait Foo {
///     fn bar() -> impl Deref<Target = impl Sized>;
///              // ^- RPITIT #1        ^- RPITIT #2
/// }
///
/// impl Foo for () {
///     fn bar() -> Box<String> { Box::new(String::new()) }
/// }
/// ```
///
/// The hidden types for the RPITITs in `bar` would be inferred to:
///     * `impl Deref` (RPITIT #1) = `Box<String>`
///     * `impl Sized` (RPITIT #2) = `String`
///
/// The relationship between these two types is straightforward in this case, but
/// may be more tenuously connected via other `impl`s and normalization rules for
/// cases of more complicated nested RPITITs.
#[instrument(skip(tcx), level = "debug", ret)]
pub(super) fn collect_return_position_impl_trait_in_trait_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m_def_id: LocalDefId,
) -> Result<&'tcx DefIdMap<ty::EarlyBinder<'tcx, Ty<'tcx>>>, ErrorGuaranteed> {
    let impl_m = tcx.opt_associated_item(impl_m_def_id.to_def_id()).unwrap();
    let trait_m = tcx.opt_associated_item(impl_m.trait_item_def_id.unwrap()).unwrap();
    let impl_trait_ref =
        tcx.impl_trait_ref(impl_m.impl_container(tcx).unwrap()).unwrap().instantiate_identity();
    // First, check a few of the same things as `compare_impl_method`,
    // just so we don't ICE during instantiation later.
    check_method_is_structurally_compatible(tcx, impl_m, trait_m, impl_trait_ref, true)?;

    let impl_m_hir_id = tcx.local_def_id_to_hir_id(impl_m_def_id);
    let return_span = tcx.hir_fn_decl_by_hir_id(impl_m_hir_id).unwrap().output.span();
    let cause = ObligationCause::new(
        return_span,
        impl_m_def_id,
        ObligationCauseCode::CompareImplItem {
            impl_item_def_id: impl_m_def_id,
            trait_item_def_id: trait_m.def_id,
            kind: impl_m.kind,
        },
    );

    // Create mapping from trait to impl (i.e. impl trait header + impl method identity args).
    let trait_to_impl_args = GenericArgs::identity_for_item(tcx, impl_m.def_id).rebase_onto(
        tcx,
        impl_m.container_id(tcx),
        impl_trait_ref.args,
    );

    let hybrid_preds = tcx
        .predicates_of(impl_m.container_id(tcx))
        .instantiate_identity(tcx)
        .into_iter()
        .chain(tcx.predicates_of(trait_m.def_id).instantiate_own(tcx, trait_to_impl_args))
        .map(|(clause, _)| clause);
    let param_env = ty::ParamEnv::new(tcx.mk_clauses_from_iter(hybrid_preds));
    let param_env = traits::normalize_param_env_or_error(
        tcx,
        param_env,
        ObligationCause::misc(tcx.def_span(impl_m_def_id), impl_m_def_id),
    );

    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);

    // Check that the where clauses of the impl are satisfied by the hybrid param env.
    // You might ask -- what does this have to do with RPITIT inference? Nothing.
    // We check these because if the where clauses of the signatures do not match
    // up, then we don't want to give spurious other errors that point at the RPITITs.
    // They're not necessary to check, though, because we already check them in
    // `compare_method_predicate_entailment`.
    let impl_m_own_bounds = tcx.predicates_of(impl_m_def_id).instantiate_own_identity();
    for (predicate, span) in impl_m_own_bounds {
        let normalize_cause = traits::ObligationCause::misc(span, impl_m_def_id);
        let predicate = ocx.normalize(&normalize_cause, param_env, predicate);

        let cause = ObligationCause::new(
            span,
            impl_m_def_id,
            ObligationCauseCode::CompareImplItem {
                impl_item_def_id: impl_m_def_id,
                trait_item_def_id: trait_m.def_id,
                kind: impl_m.kind,
            },
        );
        ocx.register_obligation(traits::Obligation::new(tcx, cause, param_env, predicate));
    }

    // Normalize the impl signature with fresh variables for lifetime inference.
    let misc_cause = ObligationCause::misc(return_span, impl_m_def_id);
    let impl_sig = ocx.normalize(
        &misc_cause,
        param_env,
        infcx.instantiate_binder_with_fresh_vars(
            return_span,
            BoundRegionConversionTime::HigherRankedType,
            tcx.fn_sig(impl_m.def_id).instantiate_identity(),
        ),
    );
    impl_sig.error_reported()?;
    let impl_return_ty = impl_sig.output();

    // Normalize the trait signature with liberated bound vars, passing it through
    // the ImplTraitInTraitCollector, which gathers all of the RPITITs and replaces
    // them with inference variables.
    // We will use these inference variables to collect the hidden types of RPITITs.
    let mut collector = ImplTraitInTraitCollector::new(&ocx, return_span, param_env, impl_m_def_id);
    let unnormalized_trait_sig = tcx
        .liberate_late_bound_regions(
            impl_m.def_id,
            tcx.fn_sig(trait_m.def_id).instantiate(tcx, trait_to_impl_args),
        )
        .fold_with(&mut collector);

    let trait_sig = ocx.normalize(&misc_cause, param_env, unnormalized_trait_sig);
    trait_sig.error_reported()?;
    let trait_return_ty = trait_sig.output();

    // RPITITs are allowed to use the implied predicates of the method that
    // defines them. This is because we want code like:
    // ```
    // trait Foo {
    //     fn test<'a, T>(_: &'a T) -> impl Sized;
    // }
    // impl Foo for () {
    //     fn test<'a, T>(x: &'a T) -> &'a T { x }
    // }
    // ```
    // .. to compile. However, since we use both the normalized and unnormalized
    // inputs and outputs from the instantiated trait signature, we will end up
    // seeing the hidden type of an RPIT in the signature itself. Naively, this
    // means that we will use the hidden type to imply the hidden type's own
    // well-formedness.
    //
    // To avoid this, we replace the infer vars used for hidden type inference
    // with placeholders, which imply nothing about outlives bounds, and then
    // prove below that the hidden types are well formed.
    let universe = infcx.create_next_universe();
    let mut idx = ty::BoundVar::ZERO;
    let mapping: FxIndexMap<_, _> = collector
        .types
        .iter()
        .map(|(_, &(ty, _))| {
            assert!(
                infcx.resolve_vars_if_possible(ty) == ty && ty.is_ty_var(),
                "{ty:?} should not have been constrained via normalization",
                ty = infcx.resolve_vars_if_possible(ty)
            );
            idx += 1;
            (
                ty,
                Ty::new_placeholder(
                    tcx,
                    ty::Placeholder {
                        universe,
                        bound: ty::BoundTy { var: idx, kind: ty::BoundTyKind::Anon },
                    },
                ),
            )
        })
        .collect();
    let mut type_mapper = BottomUpFolder {
        tcx,
        ty_op: |ty| *mapping.get(&ty).unwrap_or(&ty),
        lt_op: |lt| lt,
        ct_op: |ct| ct,
    };
    let wf_tys = FxIndexSet::from_iter(
        unnormalized_trait_sig
            .inputs_and_output
            .iter()
            .chain(trait_sig.inputs_and_output.iter())
            .map(|ty| ty.fold_with(&mut type_mapper)),
    );

    match ocx.eq(&cause, param_env, trait_return_ty, impl_return_ty) {
        Ok(()) => {}
        Err(terr) => {
            let mut diag = struct_span_code_err!(
                tcx.dcx(),
                cause.span,
                E0053,
                "method `{}` has an incompatible return type for trait",
                trait_m.name()
            );
            infcx.err_ctxt().note_type_err(
                &mut diag,
                &cause,
                tcx.hir_get_if_local(impl_m.def_id)
                    .and_then(|node| node.fn_decl())
                    .map(|decl| (decl.output.span(), Cow::from("return type in trait"), false)),
                Some(param_env.and(infer::ValuePairs::Terms(ExpectedFound {
                    expected: trait_return_ty.into(),
                    found: impl_return_ty.into(),
                }))),
                terr,
                false,
                None,
            );
            return Err(diag.emit());
        }
    }

    debug!(?trait_sig, ?impl_sig, "equating function signatures");

    // Unify the whole function signature. We need to do this to fully infer
    // the lifetimes of the return type, but do this after unifying just the
    // return types, since we want to avoid duplicating errors from
    // `compare_method_predicate_entailment`.
    match ocx.eq(&cause, param_env, trait_sig, impl_sig) {
        Ok(()) => {}
        Err(terr) => {
            // This function gets called during `compare_method_predicate_entailment` when normalizing a
            // signature that contains RPITIT. When the method signatures don't match, we have to
            // emit an error now because `compare_method_predicate_entailment` will not report the error
            // when normalization fails.
            let emitted = report_trait_method_mismatch(
                infcx,
                cause,
                param_env,
                terr,
                (trait_m, trait_sig),
                (impl_m, impl_sig),
                impl_trait_ref,
            );
            return Err(emitted);
        }
    }

    if !unnormalized_trait_sig.output().references_error() && collector.types.is_empty() {
        tcx.dcx().delayed_bug(
            "expect >0 RPITITs in call to `collect_return_position_impl_trait_in_trait_tys`",
        );
    }

    // FIXME: This has the same issue as #108544, but since this isn't breaking
    // existing code, I'm not particularly inclined to do the same hack as above
    // where we process wf obligations manually. This can be fixed in a forward-
    // compatible way later.
    let collected_types = collector.types;
    for (_, &(ty, _)) in &collected_types {
        ocx.register_obligation(traits::Obligation::new(
            tcx,
            misc_cause.clone(),
            param_env,
            ty::ClauseKind::WellFormed(ty.into()),
        ));
    }

    // Check that all obligations are satisfied by the implementation's
    // RPITs.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        if let Err(guar) = try_report_async_mismatch(tcx, infcx, &errors, trait_m, impl_m, impl_sig)
        {
            return Err(guar);
        }

        let guar = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(guar);
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters.
    ocx.resolve_regions_and_report_errors(impl_m_def_id, param_env, wf_tys)?;

    let mut remapped_types = DefIdMap::default();
    for (def_id, (ty, args)) in collected_types {
        match infcx.fully_resolve(ty) {
            Ok(ty) => {
                // `ty` contains free regions that we created earlier while liberating the
                // trait fn signature. However, projection normalization expects `ty` to
                // contains `def_id`'s early-bound regions.
                let id_args = GenericArgs::identity_for_item(tcx, def_id);
                debug!(?id_args, ?args);
                let map: FxIndexMap<_, _> = std::iter::zip(args, id_args)
                    .skip(tcx.generics_of(trait_m.def_id).count())
                    .filter_map(|(a, b)| Some((a.as_region()?, b.as_region()?)))
                    .collect();
                debug!(?map);

                // NOTE(compiler-errors): RPITITs, like all other RPITs, have early-bound
                // region args that are synthesized during AST lowering. These are args
                // that are appended to the parent args (trait and trait method). However,
                // we're trying to infer the uninstantiated type value of the RPITIT inside
                // the *impl*, so we can later use the impl's method args to normalize
                // an RPITIT to a concrete type (`confirm_impl_trait_in_trait_candidate`).
                //
                // Due to the design of RPITITs, during AST lowering, we have no idea that
                // an impl method corresponds to a trait method with RPITITs in it. Therefore,
                // we don't have a list of early-bound region args for the RPITIT in the impl.
                // Since early region parameters are index-based, we can't just rebase these
                // (trait method) early-bound region args onto the impl, and there's no
                // guarantee that the indices from the trait args and impl args line up.
                // So to fix this, we subtract the number of trait args and add the number of
                // impl args to *renumber* these early-bound regions to their corresponding
                // indices in the impl's generic parameters list.
                //
                // Also, we only need to account for a difference in trait and impl args,
                // since we previously enforce that the trait method and impl method have the
                // same generics.
                let num_trait_args = impl_trait_ref.args.len();
                let num_impl_args = tcx.generics_of(impl_m.container_id(tcx)).own_params.len();
                let ty = match ty.try_fold_with(&mut RemapHiddenTyRegions {
                    tcx,
                    map,
                    num_trait_args,
                    num_impl_args,
                    def_id,
                    impl_m_def_id: impl_m.def_id,
                    ty,
                    return_span,
                }) {
                    Ok(ty) => ty,
                    Err(guar) => Ty::new_error(tcx, guar),
                };
                remapped_types.insert(def_id, ty::EarlyBinder::bind(ty));
            }
            Err(err) => {
                // This code path is not reached in any tests, but may be
                // reachable. If this is triggered, it should be converted to
                // `span_delayed_bug` and the triggering case turned into a
                // test.
                tcx.dcx()
                    .span_bug(return_span, format!("could not fully resolve: {ty} => {err:?}"));
            }
        }
    }

    // We may not collect all RPITITs that we see in the HIR for a trait signature
    // because an RPITIT was located within a missing item. Like if we have a sig
    // returning `-> Missing<impl Sized>`, that gets converted to `-> {type error}`,
    // and when walking through the signature we end up never collecting the def id
    // of the `impl Sized`. Insert that here, so we don't ICE later.
    for assoc_item in tcx.associated_types_for_impl_traits_in_associated_fn(trait_m.def_id) {
        if !remapped_types.contains_key(assoc_item) {
            remapped_types.insert(
                *assoc_item,
                ty::EarlyBinder::bind(Ty::new_error_with_message(
                    tcx,
                    return_span,
                    "missing synthetic item for RPITIT",
                )),
            );
        }
    }

    Ok(&*tcx.arena.alloc(remapped_types))
}

struct ImplTraitInTraitCollector<'a, 'tcx, E> {
    ocx: &'a ObligationCtxt<'a, 'tcx, E>,
    types: FxIndexMap<DefId, (Ty<'tcx>, ty::GenericArgsRef<'tcx>)>,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
}

impl<'a, 'tcx, E> ImplTraitInTraitCollector<'a, 'tcx, E>
where
    E: 'tcx,
{
    fn new(
        ocx: &'a ObligationCtxt<'a, 'tcx, E>,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
    ) -> Self {
        ImplTraitInTraitCollector { ocx, types: FxIndexMap::default(), span, param_env, body_id }
    }
}

impl<'tcx, E> TypeFolder<TyCtxt<'tcx>> for ImplTraitInTraitCollector<'_, 'tcx, E>
where
    E: 'tcx,
{
    fn cx(&self) -> TyCtxt<'tcx> {
        self.ocx.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, proj) = ty.kind()
            && self.cx().is_impl_trait_in_trait(proj.def_id)
        {
            if let Some((ty, _)) = self.types.get(&proj.def_id) {
                return *ty;
            }
            //FIXME(RPITIT): Deny nested RPITIT in args too
            if proj.args.has_escaping_bound_vars() {
                bug!("FIXME(RPITIT): error here");
            }
            // Replace with infer var
            let infer_ty = self.ocx.infcx.next_ty_var(self.span);
            self.types.insert(proj.def_id, (infer_ty, proj.args));
            // Recurse into bounds
            for (pred, pred_span) in self
                .cx()
                .explicit_item_bounds(proj.def_id)
                .iter_instantiated_copied(self.cx(), proj.args)
            {
                let pred = pred.fold_with(self);
                let pred = self.ocx.normalize(
                    &ObligationCause::misc(self.span, self.body_id),
                    self.param_env,
                    pred,
                );

                self.ocx.register_obligation(traits::Obligation::new(
                    self.cx(),
                    ObligationCause::new(
                        self.span,
                        self.body_id,
                        ObligationCauseCode::WhereClause(proj.def_id, pred_span),
                    ),
                    self.param_env,
                    pred,
                ));
            }
            infer_ty
        } else {
            ty.super_fold_with(self)
        }
    }
}

struct RemapHiddenTyRegions<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// Map from early/late params of the impl to identity regions of the RPITIT (GAT)
    /// in the trait.
    map: FxIndexMap<ty::Region<'tcx>, ty::Region<'tcx>>,
    num_trait_args: usize,
    num_impl_args: usize,
    /// Def id of the RPITIT (GAT) in the *trait*.
    def_id: DefId,
    /// Def id of the impl method which owns the opaque hidden type we're remapping.
    impl_m_def_id: DefId,
    /// The hidden type we're remapping. Useful for diagnostics.
    ty: Ty<'tcx>,
    /// Span of the return type. Useful for diagnostics.
    return_span: Span,
}

impl<'tcx> ty::FallibleTypeFolder<TyCtxt<'tcx>> for RemapHiddenTyRegions<'tcx> {
    type Error = ErrorGuaranteed;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn try_fold_region(
        &mut self,
        region: ty::Region<'tcx>,
    ) -> Result<ty::Region<'tcx>, Self::Error> {
        match region.kind() {
            // Never remap bound regions or `'static`
            ty::ReBound(..) | ty::ReStatic | ty::ReError(_) => return Ok(region),
            // We always remap liberated late-bound regions from the function.
            ty::ReLateParam(_) => {}
            // Remap early-bound regions as long as they don't come from the `impl` itself,
            // in which case we don't really need to renumber them.
            ty::ReEarlyParam(ebr) => {
                if ebr.index as usize >= self.num_impl_args {
                    // Remap
                } else {
                    return Ok(region);
                }
            }
            ty::ReVar(_) | ty::RePlaceholder(_) | ty::ReErased => unreachable!(
                "should not have leaked vars or placeholders into hidden type of RPITIT"
            ),
        }

        let e = if let Some(id_region) = self.map.get(&region) {
            if let ty::ReEarlyParam(e) = id_region.kind() {
                e
            } else {
                bug!(
                    "expected to map region {region} to early-bound identity region, but got {id_region}"
                );
            }
        } else {
            let guar = match region.opt_param_def_id(self.tcx, self.impl_m_def_id) {
                Some(def_id) => {
                    let return_span = if let ty::Alias(ty::Opaque, opaque_ty) = self.ty.kind() {
                        self.tcx.def_span(opaque_ty.def_id)
                    } else {
                        self.return_span
                    };
                    self.tcx
                        .dcx()
                        .struct_span_err(
                            return_span,
                            "return type captures more lifetimes than trait definition",
                        )
                        .with_span_label(self.tcx.def_span(def_id), "this lifetime was captured")
                        .with_span_note(
                            self.tcx.def_span(self.def_id),
                            "hidden type must only reference lifetimes captured by this impl trait",
                        )
                        .with_note(format!("hidden type inferred to be `{}`", self.ty))
                        .emit()
                }
                None => {
                    // This code path is not reached in any tests, but may be
                    // reachable. If this is triggered, it should be converted
                    // to `delayed_bug` and the triggering case turned into a
                    // test.
                    self.tcx.dcx().bug("should've been able to remap region");
                }
            };
            return Err(guar);
        };

        Ok(ty::Region::new_early_param(
            self.tcx,
            ty::EarlyParamRegion {
                name: e.name,
                index: (e.index as usize - self.num_trait_args + self.num_impl_args) as u32,
            },
        ))
    }
}

/// Gets the string for an explicit self declaration, e.g. "self", "&self",
/// etc.
fn get_self_string<'tcx, P>(self_arg_ty: Ty<'tcx>, is_self_ty: P) -> String
where
    P: Fn(Ty<'tcx>) -> bool,
{
    if is_self_ty(self_arg_ty) {
        "self".to_owned()
    } else if let ty::Ref(_, ty, mutbl) = self_arg_ty.kind()
        && is_self_ty(*ty)
    {
        match mutbl {
            hir::Mutability::Not => "&self".to_owned(),
            hir::Mutability::Mut => "&mut self".to_owned(),
        }
    } else {
        format!("self: {self_arg_ty}")
    }
}

fn report_trait_method_mismatch<'tcx>(
    infcx: &InferCtxt<'tcx>,
    mut cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    terr: TypeError<'tcx>,
    (trait_m, trait_sig): (ty::AssocItem, ty::FnSig<'tcx>),
    (impl_m, impl_sig): (ty::AssocItem, ty::FnSig<'tcx>),
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> ErrorGuaranteed {
    let tcx = infcx.tcx;
    let (impl_err_span, trait_err_span) =
        extract_spans_for_error_reporting(infcx, terr, &cause, impl_m, trait_m);

    let mut diag = struct_span_code_err!(
        tcx.dcx(),
        impl_err_span,
        E0053,
        "method `{}` has an incompatible type for trait",
        trait_m.name()
    );
    match &terr {
        TypeError::ArgumentMutability(0) | TypeError::ArgumentSorts(_, 0)
            if trait_m.is_method() =>
        {
            let ty = trait_sig.inputs()[0];
            let sugg = get_self_string(ty, |ty| ty == impl_trait_ref.self_ty());

            // When the `impl` receiver is an arbitrary self type, like `self: Box<Self>`, the
            // span points only at the type `Box<Self`>, but we want to cover the whole
            // argument pattern and type.
            let (sig, body) = tcx.hir_expect_impl_item(impl_m.def_id.expect_local()).expect_fn();
            let span = tcx
                .hir_body_param_idents(body)
                .zip(sig.decl.inputs.iter())
                .map(|(param_ident, ty)| {
                    if let Some(param_ident) = param_ident {
                        param_ident.span.to(ty.span)
                    } else {
                        ty.span
                    }
                })
                .next()
                .unwrap_or(impl_err_span);

            diag.span_suggestion_verbose(
                span,
                "change the self-receiver type to match the trait",
                sugg,
                Applicability::MachineApplicable,
            );
        }
        TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(_, i) => {
            if trait_sig.inputs().len() == *i {
                // Suggestion to change output type. We do not suggest in `async` functions
                // to avoid complex logic or incorrect output.
                if let ImplItemKind::Fn(sig, _) =
                    &tcx.hir_expect_impl_item(impl_m.def_id.expect_local()).kind
                    && !sig.header.asyncness.is_async()
                {
                    let msg = "change the output type to match the trait";
                    let ap = Applicability::MachineApplicable;
                    match sig.decl.output {
                        hir::FnRetTy::DefaultReturn(sp) => {
                            let sugg = format!(" -> {}", trait_sig.output());
                            diag.span_suggestion_verbose(sp, msg, sugg, ap);
                        }
                        hir::FnRetTy::Return(hir_ty) => {
                            let sugg = trait_sig.output();
                            diag.span_suggestion_verbose(hir_ty.span, msg, sugg, ap);
                        }
                    };
                };
            } else if let Some(trait_ty) = trait_sig.inputs().get(*i) {
                diag.span_suggestion_verbose(
                    impl_err_span,
                    "change the parameter type to match the trait",
                    trait_ty,
                    Applicability::MachineApplicable,
                );
            }
        }
        _ => {}
    }

    cause.span = impl_err_span;
    infcx.err_ctxt().note_type_err(
        &mut diag,
        &cause,
        trait_err_span.map(|sp| (sp, Cow::from("type in trait"), false)),
        Some(param_env.and(infer::ValuePairs::PolySigs(ExpectedFound {
            expected: ty::Binder::dummy(trait_sig),
            found: ty::Binder::dummy(impl_sig),
        }))),
        terr,
        false,
        None,
    );

    diag.emit()
}

fn check_region_bounds_on_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    let impl_generics = tcx.generics_of(impl_m.def_id);
    let impl_params = impl_generics.own_counts().lifetimes;

    let trait_generics = tcx.generics_of(trait_m.def_id);
    let trait_params = trait_generics.own_counts().lifetimes;

    debug!(?trait_generics, ?impl_generics);

    // Must have same number of early-bound lifetime parameters.
    // Unfortunately, if the user screws up the bounds, then this
    // will change classification between early and late. E.g.,
    // if in trait we have `<'a,'b:'a>`, and in impl we just have
    // `<'a,'b>`, then we have 2 early-bound lifetime parameters
    // in trait but 0 in the impl. But if we report "expected 2
    // but found 0" it's confusing, because it looks like there
    // are zero. Since I don't quite know how to phrase things at
    // the moment, give a kind of vague error message.
    if trait_params == impl_params {
        return Ok(());
    }

    if !delay && let Some(guar) = check_region_late_boundedness(tcx, impl_m, trait_m) {
        return Err(guar);
    }

    let span = tcx
        .hir_get_generics(impl_m.def_id.expect_local())
        .expect("expected impl item to have generics or else we can't compare them")
        .span;

    let mut generics_span = None;
    let mut bounds_span = vec![];
    let mut where_span = None;

    if let Some(trait_node) = tcx.hir_get_if_local(trait_m.def_id)
        && let Some(trait_generics) = trait_node.generics()
    {
        generics_span = Some(trait_generics.span);
        // FIXME: we could potentially look at the impl's bounds to not point at bounds that
        // *are* present in the impl.
        for p in trait_generics.predicates {
            match p.kind {
                hir::WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                    bounds,
                    ..
                })
                | hir::WherePredicateKind::RegionPredicate(hir::WhereRegionPredicate {
                    bounds,
                    ..
                }) => {
                    for b in *bounds {
                        if let hir::GenericBound::Outlives(lt) = b {
                            bounds_span.push(lt.ident.span);
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(impl_node) = tcx.hir_get_if_local(impl_m.def_id)
            && let Some(impl_generics) = impl_node.generics()
        {
            let mut impl_bounds = 0;
            for p in impl_generics.predicates {
                match p.kind {
                    hir::WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                        bounds,
                        ..
                    })
                    | hir::WherePredicateKind::RegionPredicate(hir::WhereRegionPredicate {
                        bounds,
                        ..
                    }) => {
                        for b in *bounds {
                            if let hir::GenericBound::Outlives(_) = b {
                                impl_bounds += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
            if impl_bounds == bounds_span.len() {
                bounds_span = vec![];
            } else if impl_generics.has_where_clause_predicates {
                where_span = Some(impl_generics.where_clause_span);
            }
        }
    }

    let reported = tcx
        .dcx()
        .create_err(LifetimesOrBoundsMismatchOnTrait {
            span,
            item_kind: impl_m.descr(),
            ident: impl_m.ident(tcx),
            generics_span,
            bounds_span,
            where_span,
        })
        .emit_unless(delay);

    Err(reported)
}

#[allow(unused)]
enum LateEarlyMismatch<'tcx> {
    EarlyInImpl(DefId, DefId, ty::Region<'tcx>),
    LateInImpl(DefId, DefId, ty::Region<'tcx>),
}

fn check_region_late_boundedness<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
) -> Option<ErrorGuaranteed> {
    if !impl_m.is_fn() {
        return None;
    }

    let (infcx, param_env) = tcx
        .infer_ctxt()
        .build_with_typing_env(ty::TypingEnv::non_body_analysis(tcx, impl_m.def_id));

    let impl_m_args = infcx.fresh_args_for_item(DUMMY_SP, impl_m.def_id);
    let impl_m_sig = tcx.fn_sig(impl_m.def_id).instantiate(tcx, impl_m_args);
    let impl_m_sig = tcx.liberate_late_bound_regions(impl_m.def_id, impl_m_sig);

    let trait_m_args = infcx.fresh_args_for_item(DUMMY_SP, trait_m.def_id);
    let trait_m_sig = tcx.fn_sig(trait_m.def_id).instantiate(tcx, trait_m_args);
    let trait_m_sig = tcx.liberate_late_bound_regions(impl_m.def_id, trait_m_sig);

    let ocx = ObligationCtxt::new(&infcx);

    // Equate the signatures so that we can infer whether a late-bound param was present where
    // an early-bound param was expected, since we replace the late-bound lifetimes with
    // `ReLateParam`, and early-bound lifetimes with infer vars, so the early-bound args will
    // resolve to `ReLateParam` if there is a mismatch.
    let Ok(()) = ocx.eq(
        &ObligationCause::dummy(),
        param_env,
        ty::Binder::dummy(trait_m_sig),
        ty::Binder::dummy(impl_m_sig),
    ) else {
        return None;
    };

    let errors = ocx.select_where_possible();
    if !errors.is_empty() {
        return None;
    }

    let mut mismatched = vec![];

    let impl_generics = tcx.generics_of(impl_m.def_id);
    for (id_arg, arg) in
        std::iter::zip(ty::GenericArgs::identity_for_item(tcx, impl_m.def_id), impl_m_args)
    {
        if let ty::GenericArgKind::Lifetime(r) = arg.kind()
            && let ty::ReVar(vid) = r.kind()
            && let r = infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(tcx, vid)
            && let ty::ReLateParam(ty::LateParamRegion {
                kind: ty::LateParamRegionKind::Named(trait_param_def_id, _),
                ..
            }) = r.kind()
            && let ty::ReEarlyParam(ebr) = id_arg.expect_region().kind()
        {
            mismatched.push(LateEarlyMismatch::EarlyInImpl(
                impl_generics.region_param(ebr, tcx).def_id,
                trait_param_def_id,
                id_arg.expect_region(),
            ));
        }
    }

    let trait_generics = tcx.generics_of(trait_m.def_id);
    for (id_arg, arg) in
        std::iter::zip(ty::GenericArgs::identity_for_item(tcx, trait_m.def_id), trait_m_args)
    {
        if let ty::GenericArgKind::Lifetime(r) = arg.kind()
            && let ty::ReVar(vid) = r.kind()
            && let r = infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(tcx, vid)
            && let ty::ReLateParam(ty::LateParamRegion {
                kind: ty::LateParamRegionKind::Named(impl_param_def_id, _),
                ..
            }) = r.kind()
            && let ty::ReEarlyParam(ebr) = id_arg.expect_region().kind()
        {
            mismatched.push(LateEarlyMismatch::LateInImpl(
                impl_param_def_id,
                trait_generics.region_param(ebr, tcx).def_id,
                id_arg.expect_region(),
            ));
        }
    }

    if mismatched.is_empty() {
        return None;
    }

    let spans: Vec<_> = mismatched
        .iter()
        .map(|param| {
            let (LateEarlyMismatch::EarlyInImpl(impl_param_def_id, ..)
            | LateEarlyMismatch::LateInImpl(impl_param_def_id, ..)) = param;
            tcx.def_span(impl_param_def_id)
        })
        .collect();

    let mut diag = tcx
        .dcx()
        .struct_span_err(spans, "lifetime parameters do not match the trait definition")
        .with_note("lifetime parameters differ in whether they are early- or late-bound")
        .with_code(E0195);
    for mismatch in mismatched {
        match mismatch {
            LateEarlyMismatch::EarlyInImpl(
                impl_param_def_id,
                trait_param_def_id,
                early_bound_region,
            ) => {
                let mut multispan = MultiSpan::from_spans(vec![
                    tcx.def_span(impl_param_def_id),
                    tcx.def_span(trait_param_def_id),
                ]);
                multispan
                    .push_span_label(tcx.def_span(tcx.parent(impl_m.def_id)), "in this impl...");
                multispan
                    .push_span_label(tcx.def_span(tcx.parent(trait_m.def_id)), "in this trait...");
                multispan.push_span_label(
                    tcx.def_span(impl_param_def_id),
                    format!("`{}` is early-bound", tcx.item_name(impl_param_def_id)),
                );
                multispan.push_span_label(
                    tcx.def_span(trait_param_def_id),
                    format!("`{}` is late-bound", tcx.item_name(trait_param_def_id)),
                );
                if let Some(span) =
                    find_region_in_predicates(tcx, impl_m.def_id, early_bound_region)
                {
                    multispan.push_span_label(
                        span,
                        format!(
                            "this lifetime bound makes `{}` early-bound",
                            tcx.item_name(impl_param_def_id)
                        ),
                    );
                }
                diag.span_note(
                    multispan,
                    format!(
                        "`{}` differs between the trait and impl",
                        tcx.item_name(impl_param_def_id)
                    ),
                );
            }
            LateEarlyMismatch::LateInImpl(
                impl_param_def_id,
                trait_param_def_id,
                early_bound_region,
            ) => {
                let mut multispan = MultiSpan::from_spans(vec![
                    tcx.def_span(impl_param_def_id),
                    tcx.def_span(trait_param_def_id),
                ]);
                multispan
                    .push_span_label(tcx.def_span(tcx.parent(impl_m.def_id)), "in this impl...");
                multispan
                    .push_span_label(tcx.def_span(tcx.parent(trait_m.def_id)), "in this trait...");
                multispan.push_span_label(
                    tcx.def_span(impl_param_def_id),
                    format!("`{}` is late-bound", tcx.item_name(impl_param_def_id)),
                );
                multispan.push_span_label(
                    tcx.def_span(trait_param_def_id),
                    format!("`{}` is early-bound", tcx.item_name(trait_param_def_id)),
                );
                if let Some(span) =
                    find_region_in_predicates(tcx, trait_m.def_id, early_bound_region)
                {
                    multispan.push_span_label(
                        span,
                        format!(
                            "this lifetime bound makes `{}` early-bound",
                            tcx.item_name(trait_param_def_id)
                        ),
                    );
                }
                diag.span_note(
                    multispan,
                    format!(
                        "`{}` differs between the trait and impl",
                        tcx.item_name(impl_param_def_id)
                    ),
                );
            }
        }
    }

    Some(diag.emit())
}

fn find_region_in_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    early_bound_region: ty::Region<'tcx>,
) -> Option<Span> {
    for (pred, span) in tcx.explicit_predicates_of(def_id).instantiate_identity(tcx) {
        if pred.visit_with(&mut FindRegion(early_bound_region)).is_break() {
            return Some(span);
        }
    }

    struct FindRegion<'tcx>(ty::Region<'tcx>);
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for FindRegion<'tcx> {
        type Result = ControlFlow<()>;
        fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
            if r == self.0 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        }
    }

    None
}

#[instrument(level = "debug", skip(infcx))]
fn extract_spans_for_error_reporting<'tcx>(
    infcx: &infer::InferCtxt<'tcx>,
    terr: TypeError<'_>,
    cause: &ObligationCause<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
) -> (Span, Option<Span>) {
    let tcx = infcx.tcx;
    let mut impl_args = {
        let (sig, _) = tcx.hir_expect_impl_item(impl_m.def_id.expect_local()).expect_fn();
        sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span()))
    };

    let trait_args = trait_m.def_id.as_local().map(|def_id| {
        let (sig, _) = tcx.hir_expect_trait_item(def_id).expect_fn();
        sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span()))
    });

    match terr {
        TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(ExpectedFound { .. }, i) => {
            (impl_args.nth(i).unwrap(), trait_args.and_then(|mut args| args.nth(i)))
        }
        _ => (cause.span, tcx.hir_span_if_local(trait_m.def_id)),
    }
}

fn compare_self_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    // Try to give more informative error messages about self typing
    // mismatches. Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter. It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.

    let self_string = |method: ty::AssocItem| {
        let untransformed_self_ty = match method.container {
            ty::AssocItemContainer::Impl => impl_trait_ref.self_ty(),
            ty::AssocItemContainer::Trait => tcx.types.self_param,
        };
        let self_arg_ty = tcx.fn_sig(method.def_id).instantiate_identity().input(0);
        let (infcx, param_env) = tcx
            .infer_ctxt()
            .build_with_typing_env(ty::TypingEnv::non_body_analysis(tcx, method.def_id));
        let self_arg_ty = tcx.liberate_late_bound_regions(method.def_id, self_arg_ty);
        let can_eq_self = |ty| infcx.can_eq(param_env, untransformed_self_ty, ty);
        get_self_string(self_arg_ty, can_eq_self)
    };

    match (trait_m.is_method(), impl_m.is_method()) {
        (false, false) | (true, true) => {}

        (false, true) => {
            let self_descr = self_string(impl_m);
            let impl_m_span = tcx.def_span(impl_m.def_id);
            let mut err = struct_span_code_err!(
                tcx.dcx(),
                impl_m_span,
                E0185,
                "method `{}` has a `{}` declaration in the impl, but not in the trait",
                trait_m.name(),
                self_descr
            );
            err.span_label(impl_m_span, format!("`{self_descr}` used in impl"));
            if let Some(span) = tcx.hir_span_if_local(trait_m.def_id) {
                err.span_label(span, format!("trait method declared without `{self_descr}`"));
            } else {
                err.note_trait_signature(trait_m.name(), trait_m.signature(tcx));
            }
            return Err(err.emit_unless(delay));
        }

        (true, false) => {
            let self_descr = self_string(trait_m);
            let impl_m_span = tcx.def_span(impl_m.def_id);
            let mut err = struct_span_code_err!(
                tcx.dcx(),
                impl_m_span,
                E0186,
                "method `{}` has a `{}` declaration in the trait, but not in the impl",
                trait_m.name(),
                self_descr
            );
            err.span_label(impl_m_span, format!("expected `{self_descr}` in impl"));
            if let Some(span) = tcx.hir_span_if_local(trait_m.def_id) {
                err.span_label(span, format!("`{self_descr}` used in trait"));
            } else {
                err.note_trait_signature(trait_m.name(), trait_m.signature(tcx));
            }

            return Err(err.emit_unless(delay));
        }
    }

    Ok(())
}

/// Checks that the number of generics on a given assoc item in a trait impl is the same
/// as the number of generics on the respective assoc item in the trait definition.
///
/// For example this code emits the errors in the following code:
/// ```rust,compile_fail
/// trait Trait {
///     fn foo();
///     type Assoc<T>;
/// }
///
/// impl Trait for () {
///     fn foo<T>() {}
///     //~^ error
///     type Assoc = u32;
///     //~^ error
/// }
/// ```
///
/// Notably this does not error on `foo<T>` implemented as `foo<const N: u8>` or
/// `foo<const N: u8>` implemented as `foo<const N: u32>`. This is handled in
/// [`compare_generic_param_kinds`]. This function also does not handle lifetime parameters
fn compare_number_of_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_: ty::AssocItem,
    trait_: ty::AssocItem,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    let trait_own_counts = tcx.generics_of(trait_.def_id).own_counts();
    let impl_own_counts = tcx.generics_of(impl_.def_id).own_counts();

    // This avoids us erroring on `foo<T>` implemented as `foo<const N: u8>` as this is implemented
    // in `compare_generic_param_kinds` which will give a nicer error message than something like:
    // "expected 1 type parameter, found 0 type parameters"
    if (trait_own_counts.types + trait_own_counts.consts)
        == (impl_own_counts.types + impl_own_counts.consts)
    {
        return Ok(());
    }

    // We never need to emit a separate error for RPITITs, since if an RPITIT
    // has mismatched type or const generic arguments, then the method that it's
    // inheriting the generics from will also have mismatched arguments, and
    // we'll report an error for that instead. Delay a bug for safety, though.
    if trait_.is_impl_trait_in_trait() {
        // FIXME: no tests trigger this. If you find example code that does
        // trigger this, please add it to the test suite.
        tcx.dcx()
            .bug("errors comparing numbers of generics of trait/impl functions were not emitted");
    }

    let matchings = [
        ("type", trait_own_counts.types, impl_own_counts.types),
        ("const", trait_own_counts.consts, impl_own_counts.consts),
    ];

    let item_kind = impl_.descr();

    let mut err_occurred = None;
    for (kind, trait_count, impl_count) in matchings {
        if impl_count != trait_count {
            let arg_spans = |item: &ty::AssocItem, generics: &hir::Generics<'_>| {
                let mut spans = generics
                    .params
                    .iter()
                    .filter(|p| match p.kind {
                        hir::GenericParamKind::Lifetime {
                            kind: hir::LifetimeParamKind::Elided(_),
                        } => {
                            // A fn can have an arbitrary number of extra elided lifetimes for the
                            // same signature.
                            !item.is_fn()
                        }
                        _ => true,
                    })
                    .map(|p| p.span)
                    .collect::<Vec<Span>>();
                if spans.is_empty() {
                    spans = vec![generics.span]
                }
                spans
            };
            let (trait_spans, impl_trait_spans) = if let Some(def_id) = trait_.def_id.as_local() {
                let trait_item = tcx.hir_expect_trait_item(def_id);
                let arg_spans: Vec<Span> = arg_spans(&trait_, trait_item.generics);
                let impl_trait_spans: Vec<Span> = trait_item
                    .generics
                    .params
                    .iter()
                    .filter_map(|p| match p.kind {
                        GenericParamKind::Type { synthetic: true, .. } => Some(p.span),
                        _ => None,
                    })
                    .collect();
                (Some(arg_spans), impl_trait_spans)
            } else {
                let trait_span = tcx.hir_span_if_local(trait_.def_id);
                (trait_span.map(|s| vec![s]), vec![])
            };

            let impl_item = tcx.hir_expect_impl_item(impl_.def_id.expect_local());
            let impl_item_impl_trait_spans: Vec<Span> = impl_item
                .generics
                .params
                .iter()
                .filter_map(|p| match p.kind {
                    GenericParamKind::Type { synthetic: true, .. } => Some(p.span),
                    _ => None,
                })
                .collect();
            let spans = arg_spans(&impl_, impl_item.generics);
            let span = spans.first().copied();

            let mut err = tcx.dcx().struct_span_err(
                spans,
                format!(
                    "{} `{}` has {} {kind} parameter{} but its trait \
                     declaration has {} {kind} parameter{}",
                    item_kind,
                    trait_.name(),
                    impl_count,
                    pluralize!(impl_count),
                    trait_count,
                    pluralize!(trait_count),
                    kind = kind,
                ),
            );
            err.code(E0049);

            let msg =
                format!("expected {trait_count} {kind} parameter{}", pluralize!(trait_count),);
            if let Some(spans) = trait_spans {
                let mut spans = spans.iter();
                if let Some(span) = spans.next() {
                    err.span_label(*span, msg);
                }
                for span in spans {
                    err.span_label(*span, "");
                }
            } else {
                err.span_label(tcx.def_span(trait_.def_id), msg);
            }

            if let Some(span) = span {
                err.span_label(
                    span,
                    format!("found {} {} parameter{}", impl_count, kind, pluralize!(impl_count),),
                );
            }

            for span in impl_trait_spans.iter().chain(impl_item_impl_trait_spans.iter()) {
                err.span_label(*span, "`impl Trait` introduces an implicit type parameter");
            }

            let reported = err.emit_unless(delay);
            err_occurred = Some(reported);
        }
    }

    if let Some(reported) = err_occurred { Err(reported) } else { Ok(()) }
}

fn compare_number_of_method_arguments<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    let impl_m_fty = tcx.fn_sig(impl_m.def_id);
    let trait_m_fty = tcx.fn_sig(trait_m.def_id);
    let trait_number_args = trait_m_fty.skip_binder().inputs().skip_binder().len();
    let impl_number_args = impl_m_fty.skip_binder().inputs().skip_binder().len();

    if trait_number_args != impl_number_args {
        let trait_span = trait_m
            .def_id
            .as_local()
            .and_then(|def_id| {
                let (trait_m_sig, _) = &tcx.hir_expect_trait_item(def_id).expect_fn();
                let pos = trait_number_args.saturating_sub(1);
                trait_m_sig.decl.inputs.get(pos).map(|arg| {
                    if pos == 0 {
                        arg.span
                    } else {
                        arg.span.with_lo(trait_m_sig.decl.inputs[0].span.lo())
                    }
                })
            })
            .or_else(|| tcx.hir_span_if_local(trait_m.def_id));

        let (impl_m_sig, _) = &tcx.hir_expect_impl_item(impl_m.def_id.expect_local()).expect_fn();
        let pos = impl_number_args.saturating_sub(1);
        let impl_span = impl_m_sig
            .decl
            .inputs
            .get(pos)
            .map(|arg| {
                if pos == 0 {
                    arg.span
                } else {
                    arg.span.with_lo(impl_m_sig.decl.inputs[0].span.lo())
                }
            })
            .unwrap_or_else(|| tcx.def_span(impl_m.def_id));

        let mut err = struct_span_code_err!(
            tcx.dcx(),
            impl_span,
            E0050,
            "method `{}` has {} but the declaration in trait `{}` has {}",
            trait_m.name(),
            potentially_plural_count(impl_number_args, "parameter"),
            tcx.def_path_str(trait_m.def_id),
            trait_number_args
        );

        if let Some(trait_span) = trait_span {
            err.span_label(
                trait_span,
                format!(
                    "trait requires {}",
                    potentially_plural_count(trait_number_args, "parameter")
                ),
            );
        } else {
            err.note_trait_signature(trait_m.name(), trait_m.signature(tcx));
        }

        err.span_label(
            impl_span,
            format!(
                "expected {}, found {}",
                potentially_plural_count(trait_number_args, "parameter"),
                impl_number_args
            ),
        );

        return Err(err.emit_unless(delay));
    }

    Ok(())
}

fn compare_synthetic_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_m: ty::AssocItem,
    trait_m: ty::AssocItem,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    // FIXME(chrisvittal) Clean up this function, list of FIXME items:
    //     1. Better messages for the span labels
    //     2. Explanation as to what is going on
    // If we get here, we already have the same number of generics, so the zip will
    // be okay.
    let mut error_found = None;
    let impl_m_generics = tcx.generics_of(impl_m.def_id);
    let trait_m_generics = tcx.generics_of(trait_m.def_id);
    let impl_m_type_params =
        impl_m_generics.own_params.iter().filter_map(|param| match param.kind {
            GenericParamDefKind::Type { synthetic, .. } => Some((param.def_id, synthetic)),
            GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => None,
        });
    let trait_m_type_params =
        trait_m_generics.own_params.iter().filter_map(|param| match param.kind {
            GenericParamDefKind::Type { synthetic, .. } => Some((param.def_id, synthetic)),
            GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => None,
        });
    for ((impl_def_id, impl_synthetic), (trait_def_id, trait_synthetic)) in
        iter::zip(impl_m_type_params, trait_m_type_params)
    {
        if impl_synthetic != trait_synthetic {
            let impl_def_id = impl_def_id.expect_local();
            let impl_span = tcx.def_span(impl_def_id);
            let trait_span = tcx.def_span(trait_def_id);
            let mut err = struct_span_code_err!(
                tcx.dcx(),
                impl_span,
                E0643,
                "method `{}` has incompatible signature for trait",
                trait_m.name()
            );
            err.span_label(trait_span, "declaration in trait here");
            if impl_synthetic {
                // The case where the impl method uses `impl Trait` but the trait method uses
                // explicit generics
                err.span_label(impl_span, "expected generic parameter, found `impl Trait`");
                let _: Option<_> = try {
                    // try taking the name from the trait impl
                    // FIXME: this is obviously suboptimal since the name can already be used
                    // as another generic argument
                    let new_name = tcx.opt_item_name(trait_def_id)?;
                    let trait_m = trait_m.def_id.as_local()?;
                    let trait_m = tcx.hir_expect_trait_item(trait_m);

                    let impl_m = impl_m.def_id.as_local()?;
                    let impl_m = tcx.hir_expect_impl_item(impl_m);

                    // in case there are no generics, take the spot between the function name
                    // and the opening paren of the argument list
                    let new_generics_span = tcx.def_ident_span(impl_def_id)?.shrink_to_hi();
                    // in case there are generics, just replace them
                    let generics_span = impl_m.generics.span.substitute_dummy(new_generics_span);
                    // replace with the generics from the trait
                    let new_generics =
                        tcx.sess.source_map().span_to_snippet(trait_m.generics.span).ok()?;

                    err.multipart_suggestion(
                        "try changing the `impl Trait` argument to a generic parameter",
                        vec![
                            // replace `impl Trait` with `T`
                            (impl_span, new_name.to_string()),
                            // replace impl method generics with trait method generics
                            // This isn't quite right, as users might have changed the names
                            // of the generics, but it works for the common case
                            (generics_span, new_generics),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                };
            } else {
                // The case where the trait method uses `impl Trait`, but the impl method uses
                // explicit generics.
                err.span_label(impl_span, "expected `impl Trait`, found generic parameter");
                let _: Option<_> = try {
                    let impl_m = impl_m.def_id.as_local()?;
                    let impl_m = tcx.hir_expect_impl_item(impl_m);
                    let (sig, _) = impl_m.expect_fn();
                    let input_tys = sig.decl.inputs;

                    struct Visitor(hir::def_id::LocalDefId);
                    impl<'v> intravisit::Visitor<'v> for Visitor {
                        type Result = ControlFlow<Span>;
                        fn visit_ty(&mut self, ty: &'v hir::Ty<'v, AmbigArg>) -> Self::Result {
                            if let hir::TyKind::Path(hir::QPath::Resolved(None, path)) = ty.kind
                                && let Res::Def(DefKind::TyParam, def_id) = path.res
                                && def_id == self.0.to_def_id()
                            {
                                ControlFlow::Break(ty.span)
                            } else {
                                intravisit::walk_ty(self, ty)
                            }
                        }
                    }

                    let span = input_tys
                        .iter()
                        .find_map(|ty| Visitor(impl_def_id).visit_ty_unambig(ty).break_value())?;

                    let bounds = impl_m.generics.bounds_for_param(impl_def_id).next()?.bounds;
                    let bounds = bounds.first()?.span().to(bounds.last()?.span());
                    let bounds = tcx.sess.source_map().span_to_snippet(bounds).ok()?;

                    err.multipart_suggestion(
                        "try removing the generic parameter and using `impl Trait` instead",
                        vec![
                            // delete generic parameters
                            (impl_m.generics.span, String::new()),
                            // replace param usage with `impl Trait`
                            (span, format!("impl {bounds}")),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                };
            }
            error_found = Some(err.emit_unless(delay));
        }
    }
    if let Some(reported) = error_found { Err(reported) } else { Ok(()) }
}

/// Checks that all parameters in the generics of a given assoc item in a trait impl have
/// the same kind as the respective generic parameter in the trait def.
///
/// For example all 4 errors in the following code are emitted here:
/// ```rust,ignore (pseudo-Rust)
/// trait Foo {
///     fn foo<const N: u8>();
///     type Bar<const N: u8>;
///     fn baz<const N: u32>();
///     type Blah<T>;
/// }
///
/// impl Foo for () {
///     fn foo<const N: u64>() {}
///     //~^ error
///     type Bar<const N: u64> = ();
///     //~^ error
///     fn baz<T>() {}
///     //~^ error
///     type Blah<const N: i64> = u32;
///     //~^ error
/// }
/// ```
///
/// This function does not handle lifetime parameters
fn compare_generic_param_kinds<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_item: ty::AssocItem,
    trait_item: ty::AssocItem,
    delay: bool,
) -> Result<(), ErrorGuaranteed> {
    assert_eq!(impl_item.as_tag(), trait_item.as_tag());

    let ty_const_params_of = |def_id| {
        tcx.generics_of(def_id).own_params.iter().filter(|param| {
            matches!(
                param.kind,
                GenericParamDefKind::Const { .. } | GenericParamDefKind::Type { .. }
            )
        })
    };

    for (param_impl, param_trait) in
        iter::zip(ty_const_params_of(impl_item.def_id), ty_const_params_of(trait_item.def_id))
    {
        use GenericParamDefKind::*;
        if match (&param_impl.kind, &param_trait.kind) {
            (Const { .. }, Const { .. })
                if tcx.type_of(param_impl.def_id) != tcx.type_of(param_trait.def_id) =>
            {
                true
            }
            (Const { .. }, Type { .. }) | (Type { .. }, Const { .. }) => true,
            // this is exhaustive so that anyone adding new generic param kinds knows
            // to make sure this error is reported for them.
            (Const { .. }, Const { .. }) | (Type { .. }, Type { .. }) => false,
            (Lifetime { .. }, _) | (_, Lifetime { .. }) => {
                bug!("lifetime params are expected to be filtered by `ty_const_params_of`")
            }
        } {
            let param_impl_span = tcx.def_span(param_impl.def_id);
            let param_trait_span = tcx.def_span(param_trait.def_id);

            let mut err = struct_span_code_err!(
                tcx.dcx(),
                param_impl_span,
                E0053,
                "{} `{}` has an incompatible generic parameter for trait `{}`",
                impl_item.descr(),
                trait_item.name(),
                &tcx.def_path_str(tcx.parent(trait_item.def_id))
            );

            let make_param_message = |prefix: &str, param: &ty::GenericParamDef| match param.kind {
                Const { .. } => {
                    format!(
                        "{} const parameter of type `{}`",
                        prefix,
                        tcx.type_of(param.def_id).instantiate_identity()
                    )
                }
                Type { .. } => format!("{prefix} type parameter"),
                Lifetime { .. } => span_bug!(
                    tcx.def_span(param.def_id),
                    "lifetime params are expected to be filtered by `ty_const_params_of`"
                ),
            };

            let trait_header_span = tcx.def_ident_span(tcx.parent(trait_item.def_id)).unwrap();
            err.span_label(trait_header_span, "");
            err.span_label(param_trait_span, make_param_message("expected", param_trait));

            let impl_header_span = tcx.def_span(tcx.parent(impl_item.def_id));
            err.span_label(impl_header_span, "");
            err.span_label(param_impl_span, make_param_message("found", param_impl));

            let reported = err.emit_unless(delay);
            return Err(reported);
        }
    }

    Ok(())
}

fn compare_impl_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_const_item: ty::AssocItem,
    trait_const_item: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    compare_number_of_generics(tcx, impl_const_item, trait_const_item, false)?;
    compare_generic_param_kinds(tcx, impl_const_item, trait_const_item, false)?;
    check_region_bounds_on_impl_item(tcx, impl_const_item, trait_const_item, false)?;
    compare_const_predicate_entailment(tcx, impl_const_item, trait_const_item, impl_trait_ref)
}

/// The equivalent of [compare_method_predicate_entailment], but for associated constants
/// instead of associated functions.
// FIXME(generic_const_items): If possible extract the common parts of `compare_{type,const}_predicate_entailment`.
#[instrument(level = "debug", skip(tcx))]
fn compare_const_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ct: ty::AssocItem,
    trait_ct: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let impl_ct_def_id = impl_ct.def_id.expect_local();
    let impl_ct_span = tcx.def_span(impl_ct_def_id);

    // The below is for the most part highly similar to the procedure
    // for methods above. It is simpler in many respects, especially
    // because we shouldn't really have to deal with lifetimes or
    // predicates. In fact some of this should probably be put into
    // shared functions because of DRY violations...
    let trait_to_impl_args = GenericArgs::identity_for_item(tcx, impl_ct.def_id).rebase_onto(
        tcx,
        impl_ct.container_id(tcx),
        impl_trait_ref.args,
    );

    // Create a parameter environment that represents the implementation's
    // associated const.
    let impl_ty = tcx.type_of(impl_ct_def_id).instantiate_identity();

    let trait_ty = tcx.type_of(trait_ct.def_id).instantiate(tcx, trait_to_impl_args);
    let code = ObligationCauseCode::CompareImplItem {
        impl_item_def_id: impl_ct_def_id,
        trait_item_def_id: trait_ct.def_id,
        kind: impl_ct.kind,
    };
    let mut cause = ObligationCause::new(impl_ct_span, impl_ct_def_id, code.clone());

    let impl_ct_predicates = tcx.predicates_of(impl_ct.def_id);
    let trait_ct_predicates = tcx.predicates_of(trait_ct.def_id);

    // The predicates declared by the impl definition, the trait and the
    // associated const in the trait are assumed.
    let impl_predicates = tcx.predicates_of(impl_ct_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate_identity(tcx).predicates;
    hybrid_preds.extend(
        trait_ct_predicates
            .instantiate_own(tcx, trait_to_impl_args)
            .map(|(predicate, _)| predicate),
    );

    let param_env = ty::ParamEnv::new(tcx.mk_clauses(&hybrid_preds));
    let param_env = traits::normalize_param_env_or_error(
        tcx,
        param_env,
        ObligationCause::misc(impl_ct_span, impl_ct_def_id),
    );

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

    let impl_ct_own_bounds = impl_ct_predicates.instantiate_own_identity();
    for (predicate, span) in impl_ct_own_bounds {
        let cause = ObligationCause::misc(span, impl_ct_def_id);
        let predicate = ocx.normalize(&cause, param_env, predicate);

        let cause = ObligationCause::new(span, impl_ct_def_id, code.clone());
        ocx.register_obligation(traits::Obligation::new(tcx, cause, param_env, predicate));
    }

    // There is no "body" here, so just pass dummy id.
    let impl_ty = ocx.normalize(&cause, param_env, impl_ty);
    debug!(?impl_ty);

    let trait_ty = ocx.normalize(&cause, param_env, trait_ty);
    debug!(?trait_ty);

    let err = ocx.sup(&cause, param_env, trait_ty, impl_ty);

    if let Err(terr) = err {
        debug!(?impl_ty, ?trait_ty);

        // Locate the Span containing just the type of the offending impl
        let (ty, _) = tcx.hir_expect_impl_item(impl_ct_def_id).expect_const();
        cause.span = ty.span;

        let mut diag = struct_span_code_err!(
            tcx.dcx(),
            cause.span,
            E0326,
            "implemented const `{}` has an incompatible type for trait",
            trait_ct.name()
        );

        let trait_c_span = trait_ct.def_id.as_local().map(|trait_ct_def_id| {
            // Add a label to the Span containing just the type of the const
            let (ty, _) = tcx.hir_expect_trait_item(trait_ct_def_id).expect_const();
            ty.span
        });

        infcx.err_ctxt().note_type_err(
            &mut diag,
            &cause,
            trait_c_span.map(|span| (span, Cow::from("type in trait"), false)),
            Some(param_env.and(infer::ValuePairs::Terms(ExpectedFound {
                expected: trait_ty.into(),
                found: impl_ty.into(),
            }))),
            terr,
            false,
            None,
        );
        return Err(diag.emit());
    };

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        return Err(infcx.err_ctxt().report_fulfillment_errors(errors));
    }

    ocx.resolve_regions_and_report_errors(impl_ct_def_id, param_env, [])
}

#[instrument(level = "debug", skip(tcx))]
fn compare_impl_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ty: ty::AssocItem,
    trait_ty: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    compare_number_of_generics(tcx, impl_ty, trait_ty, false)?;
    compare_generic_param_kinds(tcx, impl_ty, trait_ty, false)?;
    check_region_bounds_on_impl_item(tcx, impl_ty, trait_ty, false)?;
    compare_type_predicate_entailment(tcx, impl_ty, trait_ty, impl_trait_ref)?;
    check_type_bounds(tcx, trait_ty, impl_ty, impl_trait_ref)
}

/// The equivalent of [compare_method_predicate_entailment], but for associated types
/// instead of associated functions.
#[instrument(level = "debug", skip(tcx))]
fn compare_type_predicate_entailment<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ty: ty::AssocItem,
    trait_ty: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let impl_def_id = impl_ty.container_id(tcx);
    let trait_to_impl_args = GenericArgs::identity_for_item(tcx, impl_ty.def_id).rebase_onto(
        tcx,
        impl_def_id,
        impl_trait_ref.args,
    );

    let impl_ty_predicates = tcx.predicates_of(impl_ty.def_id);
    let trait_ty_predicates = tcx.predicates_of(trait_ty.def_id);

    let impl_ty_own_bounds = impl_ty_predicates.instantiate_own_identity();
    // If there are no bounds, then there are no const conditions, so no need to check that here.
    if impl_ty_own_bounds.len() == 0 {
        // Nothing to check.
        return Ok(());
    }

    // This `DefId` should be used for the `body_id` field on each
    // `ObligationCause` (and the `FnCtxt`). This is what
    // `regionck_item` expects.
    let impl_ty_def_id = impl_ty.def_id.expect_local();
    debug!(?trait_to_impl_args);

    // The predicates declared by the impl definition, the trait and the
    // associated type in the trait are assumed.
    let impl_predicates = tcx.predicates_of(impl_ty_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate_identity(tcx).predicates;
    hybrid_preds.extend(
        trait_ty_predicates
            .instantiate_own(tcx, trait_to_impl_args)
            .map(|(predicate, _)| predicate),
    );
    debug!(?hybrid_preds);

    let impl_ty_span = tcx.def_span(impl_ty_def_id);
    let normalize_cause = ObligationCause::misc(impl_ty_span, impl_ty_def_id);

    let is_conditionally_const = tcx.is_conditionally_const(impl_ty.def_id);
    if is_conditionally_const {
        // Augment the hybrid param-env with the const conditions
        // of the impl header and the trait assoc type.
        hybrid_preds.extend(
            tcx.const_conditions(impl_ty_predicates.parent.unwrap())
                .instantiate_identity(tcx)
                .into_iter()
                .chain(
                    tcx.const_conditions(trait_ty.def_id).instantiate_own(tcx, trait_to_impl_args),
                )
                .map(|(trait_ref, _)| {
                    trait_ref.to_host_effect_clause(tcx, ty::BoundConstness::Maybe)
                }),
        );
    }

    let param_env = ty::ParamEnv::new(tcx.mk_clauses(&hybrid_preds));
    let param_env = traits::normalize_param_env_or_error(tcx, param_env, normalize_cause);
    debug!(caller_bounds=?param_env.caller_bounds());

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

    for (predicate, span) in impl_ty_own_bounds {
        let cause = ObligationCause::misc(span, impl_ty_def_id);
        let predicate = ocx.normalize(&cause, param_env, predicate);

        let cause = ObligationCause::new(
            span,
            impl_ty_def_id,
            ObligationCauseCode::CompareImplItem {
                impl_item_def_id: impl_ty.def_id.expect_local(),
                trait_item_def_id: trait_ty.def_id,
                kind: impl_ty.kind,
            },
        );
        ocx.register_obligation(traits::Obligation::new(tcx, cause, param_env, predicate));
    }

    if is_conditionally_const {
        // Validate the const conditions of the impl associated type.
        let impl_ty_own_const_conditions =
            tcx.const_conditions(impl_ty.def_id).instantiate_own_identity();
        for (const_condition, span) in impl_ty_own_const_conditions {
            let normalize_cause = traits::ObligationCause::misc(span, impl_ty_def_id);
            let const_condition = ocx.normalize(&normalize_cause, param_env, const_condition);

            let cause = ObligationCause::new(
                span,
                impl_ty_def_id,
                ObligationCauseCode::CompareImplItem {
                    impl_item_def_id: impl_ty_def_id,
                    trait_item_def_id: trait_ty.def_id,
                    kind: impl_ty.kind,
                },
            );
            ocx.register_obligation(traits::Obligation::new(
                tcx,
                cause,
                param_env,
                const_condition.to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
            ));
        }
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let reported = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(reported);
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters.
    ocx.resolve_regions_and_report_errors(impl_ty_def_id, param_env, [])
}

/// Validate that `ProjectionCandidate`s created for this associated type will
/// be valid.
///
/// Usually given
///
/// trait X { type Y: Copy } impl X for T { type Y = S; }
///
/// We are able to normalize `<T as X>::Y` to `S`, and so when we check the
/// impl is well-formed we have to prove `S: Copy`.
///
/// For default associated types the normalization is not possible (the value
/// from the impl could be overridden). We also can't normalize generic
/// associated types (yet) because they contain bound parameters.
#[instrument(level = "debug", skip(tcx))]
pub(super) fn check_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ty: ty::AssocItem,
    impl_ty: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    // Avoid bogus "type annotations needed `Foo: Bar`" errors on `impl Bar for Foo` in case
    // other `Foo` impls are incoherent.
    tcx.ensure_ok().coherent_trait(impl_trait_ref.def_id)?;

    let param_env = tcx.param_env(impl_ty.def_id);
    debug!(?param_env);

    let container_id = impl_ty.container_id(tcx);
    let impl_ty_def_id = impl_ty.def_id.expect_local();
    let impl_ty_args = GenericArgs::identity_for_item(tcx, impl_ty.def_id);
    let rebased_args = impl_ty_args.rebase_onto(tcx, container_id, impl_trait_ref.args);

    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

    // A synthetic impl Trait for RPITIT desugaring or assoc type for effects desugaring has no HIR,
    // which we currently use to get the span for an impl's associated type. Instead, for these,
    // use the def_span for the synthesized  associated type.
    let impl_ty_span = if impl_ty.is_impl_trait_in_trait() {
        tcx.def_span(impl_ty_def_id)
    } else {
        match tcx.hir_node_by_def_id(impl_ty_def_id) {
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Type(_, Some(ty)),
                ..
            }) => ty.span,
            hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Type(ty), .. }) => ty.span,
            item => span_bug!(
                tcx.def_span(impl_ty_def_id),
                "cannot call `check_type_bounds` on item: {item:?}",
            ),
        }
    };
    let assumed_wf_types = ocx.assumed_wf_types_and_report_errors(param_env, impl_ty_def_id)?;

    let normalize_cause = ObligationCause::new(
        impl_ty_span,
        impl_ty_def_id,
        ObligationCauseCode::CheckAssociatedTypeBounds {
            impl_item_def_id: impl_ty.def_id.expect_local(),
            trait_item_def_id: trait_ty.def_id,
        },
    );
    let mk_cause = |span: Span| {
        let code = ObligationCauseCode::WhereClause(trait_ty.def_id, span);
        ObligationCause::new(impl_ty_span, impl_ty_def_id, code)
    };

    let mut obligations: Vec<_> = util::elaborate(
        tcx,
        tcx.explicit_item_bounds(trait_ty.def_id).iter_instantiated_copied(tcx, rebased_args).map(
            |(concrete_ty_bound, span)| {
                debug!(?concrete_ty_bound);
                traits::Obligation::new(tcx, mk_cause(span), param_env, concrete_ty_bound)
            },
        ),
    )
    .collect();

    // Only in a const implementation do we need to check that the `[const]` item bounds hold.
    if tcx.is_conditionally_const(impl_ty_def_id) {
        obligations.extend(util::elaborate(
            tcx,
            tcx.explicit_implied_const_bounds(trait_ty.def_id)
                .iter_instantiated_copied(tcx, rebased_args)
                .map(|(c, span)| {
                    traits::Obligation::new(
                        tcx,
                        mk_cause(span),
                        param_env,
                        c.to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
                    )
                }),
        ));
    }
    debug!(item_bounds=?obligations);

    // Normalize predicates with the assumption that the GAT may always normalize
    // to its definition type. This should be the param-env we use to *prove* the
    // predicate too, but we don't do that because of performance issues.
    // See <https://github.com/rust-lang/rust/pull/117542#issue-1976337685>.
    let normalize_param_env = param_env_with_gat_bounds(tcx, impl_ty, impl_trait_ref);
    for obligation in &mut obligations {
        match ocx.deeply_normalize(&normalize_cause, normalize_param_env, obligation.predicate) {
            Ok(pred) => obligation.predicate = pred,
            Err(e) => {
                return Err(infcx.err_ctxt().report_fulfillment_errors(e));
            }
        }
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    ocx.register_obligations(obligations);
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let reported = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(reported);
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters.
    ocx.resolve_regions_and_report_errors(impl_ty_def_id, param_env, assumed_wf_types)
}

/// Install projection predicates that allow GATs to project to their own
/// definition types. This is not allowed in general in cases of default
/// associated types in trait definitions, or when specialization is involved,
/// but is needed when checking these definition types actually satisfy the
/// trait bounds of the GAT.
///
/// # How it works
///
/// ```ignore (example)
/// impl<A, B> Foo<u32> for (A, B) {
///     type Bar<C> = Wrapper<A, B, C>
/// }
/// ```
///
/// - `impl_trait_ref` would be `<(A, B) as Foo<u32>>`
/// - `normalize_impl_ty_args` would be `[A, B, ^0.0]` (`^0.0` here is the bound var with db 0 and index 0)
/// - `normalize_impl_ty` would be `Wrapper<A, B, ^0.0>`
/// - `rebased_args` would be `[(A, B), u32, ^0.0]`, combining the args from
///    the *trait* with the generic associated type parameters (as bound vars).
///
/// A note regarding the use of bound vars here:
/// Imagine as an example
/// ```
/// trait Family {
///     type Member<C: Eq>;
/// }
///
/// impl Family for VecFamily {
///     type Member<C: Eq> = i32;
/// }
/// ```
/// Here, we would generate
/// ```ignore (pseudo-rust)
/// forall<C> { Normalize(<VecFamily as Family>::Member<C> => i32) }
/// ```
///
/// when we really would like to generate
/// ```ignore (pseudo-rust)
/// forall<C> { Normalize(<VecFamily as Family>::Member<C> => i32) :- Implemented(C: Eq) }
/// ```
///
/// But, this is probably fine, because although the first clause can be used with types `C` that
/// do not implement `Eq`, for it to cause some kind of problem, there would have to be a
/// `VecFamily::Member<X>` for some type `X` where `!(X: Eq)`, that appears in the value of type
/// `Member<C: Eq> = ....` That type would fail a well-formedness check that we ought to be doing
/// elsewhere, which would check that any `<T as Family>::Member<X>` meets the bounds declared in
/// the trait (notably, that `X: Eq` and `T: Family`).
fn param_env_with_gat_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_ty: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> ty::ParamEnv<'tcx> {
    let param_env = tcx.param_env(impl_ty.def_id);
    let container_id = impl_ty.container_id(tcx);
    let mut predicates = param_env.caller_bounds().to_vec();

    // for RPITITs, we should install predicates that allow us to project all
    // of the RPITITs associated with the same body. This is because checking
    // the item bounds of RPITITs often involves nested RPITITs having to prove
    // bounds about themselves.
    let impl_tys_to_install = match impl_ty.kind {
        ty::AssocKind::Type {
            data:
                ty::AssocTypeData::Rpitit(
                    ty::ImplTraitInTraitData::Impl { fn_def_id }
                    | ty::ImplTraitInTraitData::Trait { fn_def_id, .. },
                ),
        } => tcx
            .associated_types_for_impl_traits_in_associated_fn(fn_def_id)
            .iter()
            .map(|def_id| tcx.associated_item(*def_id))
            .collect(),
        _ => vec![impl_ty],
    };

    for impl_ty in impl_tys_to_install {
        let trait_ty = match impl_ty.container {
            ty::AssocItemContainer::Trait => impl_ty,
            ty::AssocItemContainer::Impl => tcx.associated_item(impl_ty.trait_item_def_id.unwrap()),
        };

        let mut bound_vars: smallvec::SmallVec<[ty::BoundVariableKind; 8]> =
            smallvec::SmallVec::with_capacity(tcx.generics_of(impl_ty.def_id).own_params.len());
        // Extend the impl's identity args with late-bound GAT vars
        let normalize_impl_ty_args = ty::GenericArgs::identity_for_item(tcx, container_id)
            .extend_to(tcx, impl_ty.def_id, |param, _| match param.kind {
                GenericParamDefKind::Type { .. } => {
                    let kind = ty::BoundTyKind::Param(param.def_id, param.name);
                    let bound_var = ty::BoundVariableKind::Ty(kind);
                    bound_vars.push(bound_var);
                    Ty::new_bound(
                        tcx,
                        ty::INNERMOST,
                        ty::BoundTy { var: ty::BoundVar::from_usize(bound_vars.len() - 1), kind },
                    )
                    .into()
                }
                GenericParamDefKind::Lifetime => {
                    let kind = ty::BoundRegionKind::Named(param.def_id, param.name);
                    let bound_var = ty::BoundVariableKind::Region(kind);
                    bound_vars.push(bound_var);
                    ty::Region::new_bound(
                        tcx,
                        ty::INNERMOST,
                        ty::BoundRegion {
                            var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                            kind,
                        },
                    )
                    .into()
                }
                GenericParamDefKind::Const { .. } => {
                    let bound_var = ty::BoundVariableKind::Const;
                    bound_vars.push(bound_var);
                    ty::Const::new_bound(
                        tcx,
                        ty::INNERMOST,
                        ty::BoundVar::from_usize(bound_vars.len() - 1),
                    )
                    .into()
                }
            });
        // When checking something like
        //
        // trait X { type Y: PartialEq<<Self as X>::Y> }
        // impl X for T { default type Y = S; }
        //
        // We will have to prove the bound S: PartialEq<<T as X>::Y>. In this case
        // we want <T as X>::Y to normalize to S. This is valid because we are
        // checking the default value specifically here. Add this equality to the
        // ParamEnv for normalization specifically.
        let normalize_impl_ty =
            tcx.type_of(impl_ty.def_id).instantiate(tcx, normalize_impl_ty_args);
        let rebased_args =
            normalize_impl_ty_args.rebase_onto(tcx, container_id, impl_trait_ref.args);
        let bound_vars = tcx.mk_bound_variable_kinds(&bound_vars);

        match normalize_impl_ty.kind() {
            ty::Alias(ty::Projection, proj)
                if proj.def_id == trait_ty.def_id && proj.args == rebased_args =>
            {
                // Don't include this predicate if the projected type is
                // exactly the same as the projection. This can occur in
                // (somewhat dubious) code like this:
                //
                // impl<T> X for T where T: X { type Y = <T as X>::Y; }
            }
            _ => predicates.push(
                ty::Binder::bind_with_vars(
                    ty::ProjectionPredicate {
                        projection_term: ty::AliasTerm::new_from_args(
                            tcx,
                            trait_ty.def_id,
                            rebased_args,
                        ),
                        term: normalize_impl_ty.into(),
                    },
                    bound_vars,
                )
                .upcast(tcx),
            ),
        };
    }

    ty::ParamEnv::new(tcx.mk_clauses(&predicates))
}

/// Manually check here that `async fn foo()` wasn't matched against `fn foo()`,
/// and extract a better error if so.
fn try_report_async_mismatch<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    errors: &[FulfillmentError<'tcx>],
    trait_m: ty::AssocItem,
    impl_m: ty::AssocItem,
    impl_sig: ty::FnSig<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    if !tcx.asyncness(trait_m.def_id).is_async() {
        return Ok(());
    }

    let ty::Alias(ty::Projection, ty::AliasTy { def_id: async_future_def_id, .. }) =
        *tcx.fn_sig(trait_m.def_id).skip_binder().skip_binder().output().kind()
    else {
        bug!("expected `async fn` to return an RPITIT");
    };

    for error in errors {
        if let ObligationCauseCode::WhereClause(def_id, _) = *error.root_obligation.cause.code()
            && def_id == async_future_def_id
            && let Some(proj) = error.root_obligation.predicate.as_projection_clause()
            && let Some(proj) = proj.no_bound_vars()
            && infcx.can_eq(
                error.root_obligation.param_env,
                proj.term.expect_type(),
                impl_sig.output(),
            )
        {
            // FIXME: We should suggest making the fn `async`, but extracting
            // the right span is a bit difficult.
            return Err(tcx.sess.dcx().emit_err(MethodShouldReturnFuture {
                span: tcx.def_span(impl_m.def_id),
                method_name: tcx.item_ident(impl_m.def_id),
                trait_item_span: tcx.hir_span_if_local(trait_m.def_id),
            }));
        }
    }

    Ok(())
}
