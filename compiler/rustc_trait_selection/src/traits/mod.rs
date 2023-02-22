//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod auto_trait;
mod chalk_fulfill;
mod coherence;
pub mod const_evaluatable;
mod engine;
pub mod error_reporting;
mod fulfill;
pub mod misc;
mod object_safety;
pub mod outlives_bounds;
mod project;
pub mod query;
mod select;
mod specialize;
mod structural_match;
mod util;
mod vtable;
pub mod wf;

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::traits::error_reporting::TypeErrCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_errors::ErrorGuaranteed;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::visit::{TypeVisitable, TypeVisitableExt};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt, TypeSuperVisitable};
use rustc_middle::ty::{InternalSubsts, SubstsRef};
use rustc_span::def_id::{DefId, CRATE_DEF_ID};
use rustc_span::Span;

use std::fmt::Debug;
use std::ops::ControlFlow;

pub use self::FulfillmentErrorCode::*;
pub use self::ImplSource::*;
pub use self::ObligationCauseCode::*;
pub use self::SelectionError::*;

pub use self::coherence::{add_placeholder_note, orphan_check, overlapping_impls};
pub use self::coherence::{OrphanCheckErr, OverlapResult};
pub use self::engine::{ObligationCtxt, TraitEngineExt};
pub use self::fulfill::{FulfillmentContext, PendingPredicateObligation};
pub use self::object_safety::astconv_object_safety_violations;
pub use self::object_safety::is_vtable_safe_method;
pub use self::object_safety::MethodViolationCode;
pub use self::object_safety::ObjectSafetyViolation;
pub use self::project::{normalize_projection_type, NormalizeExt};
pub use self::select::{EvaluationCache, SelectionCache, SelectionContext};
pub use self::select::{EvaluationResult, IntercrateAmbiguityCause, OverflowError};
pub use self::specialize::specialization_graph::FutureCompatOverlapError;
pub use self::specialize::specialization_graph::FutureCompatOverlapErrorKind;
pub use self::specialize::{specialization_graph, translate_substs, OverlapError};
pub use self::structural_match::{
    search_for_adt_const_param_violation, search_for_structural_match_violation,
};
pub use self::util::{
    elaborate_obligations, elaborate_predicates, elaborate_predicates_with_span,
    elaborate_trait_ref, elaborate_trait_refs,
};
pub use self::util::{expand_trait_aliases, TraitAliasExpander};
pub use self::util::{
    get_vtable_index_of_object_method, impl_item_is_final, predicate_for_trait_def, upcast_choices,
};
pub use self::util::{
    supertrait_def_ids, supertraits, transitive_bounds, transitive_bounds_that_define_assoc_type,
    SupertraitDefIds, Supertraits,
};

pub use self::chalk_fulfill::FulfillmentContext as ChalkFulfillmentContext;

pub use rustc_infer::traits::*;

/// Whether to skip the leak check, as part of a future compatibility warning step.
///
/// The "default" for skip-leak-check corresponds to the current
/// behavior (do not skip the leak check) -- not the behavior we are
/// transitioning into.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum SkipLeakCheck {
    Yes,
    #[default]
    No,
}

impl SkipLeakCheck {
    fn is_yes(self) -> bool {
        self == SkipLeakCheck::Yes
    }
}

/// The mode that trait queries run in.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TraitQueryMode {
    /// Standard/un-canonicalized queries get accurate
    /// spans etc. passed in and hence can do reasonable
    /// error reporting on their own.
    Standard,
    /// Canonical queries get dummy spans and hence
    /// must generally propagate errors to
    /// pre-canonicalization callsites.
    Canonical,
}

/// Creates predicate obligations from the generic bounds.
#[instrument(level = "debug", skip(cause, param_env))]
pub fn predicates_for_generics<'tcx>(
    cause: impl Fn(usize, Span) -> ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    generic_bounds: ty::InstantiatedPredicates<'tcx>,
) -> impl Iterator<Item = PredicateObligation<'tcx>> {
    generic_bounds.into_iter().enumerate().map(move |(idx, (predicate, span))| Obligation {
        cause: cause(idx, span),
        recursion_depth: 0,
        param_env,
        predicate,
    })
}

/// Determines whether the type `ty` is known to meet `bound` and
/// returns true if so. Returns false if `ty` either does not meet
/// `bound` or is not known to meet bound (note that this is
/// conservative towards *no impl*, which is the opposite of the
/// `evaluate` methods).
pub fn type_known_to_meet_bound_modulo_regions<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    def_id: DefId,
    span: Span,
) -> bool {
    let trait_ref = ty::Binder::dummy(infcx.tcx.mk_trait_ref(def_id, [ty]));
    pred_known_to_hold_modulo_regions(infcx, param_env, trait_ref.without_const(), span)
}

#[instrument(level = "debug", skip(infcx, param_env, span, pred), ret)]
fn pred_known_to_hold_modulo_regions<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pred: impl ToPredicate<'tcx> + TypeVisitable<TyCtxt<'tcx>>,
    span: Span,
) -> bool {
    let has_non_region_infer = pred.has_non_region_infer();
    let obligation = Obligation {
        param_env,
        // We can use a dummy node-id here because we won't pay any mind
        // to region obligations that arise (there shouldn't really be any
        // anyhow).
        cause: ObligationCause::misc(span, CRATE_DEF_ID),
        recursion_depth: 0,
        predicate: pred.to_predicate(infcx.tcx),
    };

    let result = infcx.predicate_must_hold_modulo_regions(&obligation);
    debug!(?result);

    if result && has_non_region_infer {
        // Because of inference "guessing", selection can sometimes claim
        // to succeed while the success requires a guess. To ensure
        // this function's result remains infallible, we must confirm
        // that guess. While imperfect, I believe this is sound.

        // FIXME(@lcnr): this function doesn't seem right.
        //
        // The handling of regions in this area of the code is terrible,
        // see issue #29149. We should be able to improve on this with
        // NLL.
        let errors = fully_solve_obligation(infcx, obligation);

        match &errors[..] {
            [] => true,
            errors => {
                debug!(?errors);
                false
            }
        }
    } else {
        result
    }
}

#[instrument(level = "debug", skip(tcx, elaborated_env))]
fn do_normalize_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: ObligationCause<'tcx>,
    elaborated_env: ty::ParamEnv<'tcx>,
    predicates: Vec<ty::Predicate<'tcx>>,
) -> Result<Vec<ty::Predicate<'tcx>>, ErrorGuaranteed> {
    let span = cause.span;
    // FIXME. We should really... do something with these region
    // obligations. But this call just continues the older
    // behavior (i.e., doesn't cause any new bugs), and it would
    // take some further refactoring to actually solve them. In
    // particular, we would have to handle implied bounds
    // properly, and that code is currently largely confined to
    // regionck (though I made some efforts to extract it
    // out). -nmatsakis
    //
    // @arielby: In any case, these obligations are checked
    // by wfcheck anyway, so I'm not sure we have to check
    // them here too, and we will remove this function when
    // we move over to lazy normalization *anyway*.
    let infcx = tcx.infer_ctxt().ignoring_regions().build();
    let predicates = match fully_normalize(&infcx, cause, elaborated_env, predicates) {
        Ok(predicates) => predicates,
        Err(errors) => {
            let reported = infcx.err_ctxt().report_fulfillment_errors(&errors, None);
            return Err(reported);
        }
    };

    debug!("do_normalize_predictes: normalized predicates = {:?}", predicates);

    // We can use the `elaborated_env` here; the region code only
    // cares about declarations like `'a: 'b`.
    let outlives_env = OutlivesEnvironment::new(elaborated_env);

    // FIXME: It's very weird that we ignore region obligations but apparently
    // still need to use `resolve_regions` as we need the resolved regions in
    // the normalized predicates.
    let errors = infcx.resolve_regions(&outlives_env);
    if !errors.is_empty() {
        tcx.sess.delay_span_bug(
            span,
            format!("failed region resolution while normalizing {elaborated_env:?}: {errors:?}"),
        );
    }

    match infcx.fully_resolve(predicates) {
        Ok(predicates) => Ok(predicates),
        Err(fixup_err) => {
            // If we encounter a fixup error, it means that some type
            // variable wound up unconstrained. I actually don't know
            // if this can happen, and I certainly don't expect it to
            // happen often, but if it did happen it probably
            // represents a legitimate failure due to some kind of
            // unconstrained variable.
            //
            // @lcnr: Let's still ICE here for now. I want a test case
            // for that.
            span_bug!(
                span,
                "inference variables in normalized parameter environment: {}",
                fixup_err
            );
        }
    }
}

// FIXME: this is gonna need to be removed ...
/// Normalizes the parameter environment, reporting errors if they occur.
#[instrument(level = "debug", skip(tcx))]
pub fn normalize_param_env_or_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    unnormalized_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
) -> ty::ParamEnv<'tcx> {
    // I'm not wild about reporting errors here; I'd prefer to
    // have the errors get reported at a defined place (e.g.,
    // during typeck). Instead I have all parameter
    // environments, in effect, going through this function
    // and hence potentially reporting errors. This ensures of
    // course that we never forget to normalize (the
    // alternative seemed like it would involve a lot of
    // manual invocations of this fn -- and then we'd have to
    // deal with the errors at each of those sites).
    //
    // In any case, in practice, typeck constructs all the
    // parameter environments once for every fn as it goes,
    // and errors will get reported then; so outside of type inference we
    // can be sure that no errors should occur.
    let mut predicates: Vec<_> =
        util::elaborate_predicates(tcx, unnormalized_env.caller_bounds().into_iter())
            .map(|obligation| obligation.predicate)
            .collect();

    debug!("normalize_param_env_or_error: elaborated-predicates={:?}", predicates);

    let elaborated_env = ty::ParamEnv::new(
        tcx.mk_predicates(&predicates),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    );

    // HACK: we are trying to normalize the param-env inside *itself*. The problem is that
    // normalization expects its param-env to be already normalized, which means we have
    // a circularity.
    //
    // The way we handle this is by normalizing the param-env inside an unnormalized version
    // of the param-env, which means that if the param-env contains unnormalized projections,
    // we'll have some normalization failures. This is unfortunate.
    //
    // Lazy normalization would basically handle this by treating just the
    // normalizing-a-trait-ref-requires-itself cycles as evaluation failures.
    //
    // Inferred outlives bounds can create a lot of `TypeOutlives` predicates for associated
    // types, so to make the situation less bad, we normalize all the predicates *but*
    // the `TypeOutlives` predicates first inside the unnormalized parameter environment, and
    // then we normalize the `TypeOutlives` bounds inside the normalized parameter environment.
    //
    // This works fairly well because trait matching does not actually care about param-env
    // TypeOutlives predicates - these are normally used by regionck.
    let outlives_predicates: Vec<_> = predicates
        .drain_filter(|predicate| {
            matches!(
                predicate.kind().skip_binder(),
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(..))
            )
        })
        .collect();

    debug!(
        "normalize_param_env_or_error: predicates=(non-outlives={:?}, outlives={:?})",
        predicates, outlives_predicates
    );
    let Ok(non_outlives_predicates) = do_normalize_predicates(
        tcx,
        cause.clone(),
        elaborated_env,
        predicates,
    ) else {
        // An unnormalized env is better than nothing.
        debug!("normalize_param_env_or_error: errored resolving non-outlives predicates");
        return elaborated_env;
    };

    debug!("normalize_param_env_or_error: non-outlives predicates={:?}", non_outlives_predicates);

    // Not sure whether it is better to include the unnormalized TypeOutlives predicates
    // here. I believe they should not matter, because we are ignoring TypeOutlives param-env
    // predicates here anyway. Keeping them here anyway because it seems safer.
    let outlives_env = non_outlives_predicates.iter().chain(&outlives_predicates).cloned();
    let outlives_env = ty::ParamEnv::new(
        tcx.mk_predicates_from_iter(outlives_env),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    );
    let Ok(outlives_predicates) = do_normalize_predicates(
        tcx,
        cause,
        outlives_env,
        outlives_predicates,
    ) else {
        // An unnormalized env is better than nothing.
        debug!("normalize_param_env_or_error: errored resolving outlives predicates");
        return elaborated_env;
    };
    debug!("normalize_param_env_or_error: outlives predicates={:?}", outlives_predicates);

    let mut predicates = non_outlives_predicates;
    predicates.extend(outlives_predicates);
    debug!("normalize_param_env_or_error: final predicates={:?}", predicates);
    ty::ParamEnv::new(
        tcx.mk_predicates(&predicates),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    )
}

/// Normalize a type and process all resulting obligations, returning any errors
#[instrument(skip_all)]
pub fn fully_normalize<'tcx, T>(
    infcx: &InferCtxt<'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: T,
) -> Result<T, Vec<FulfillmentError<'tcx>>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    let ocx = ObligationCtxt::new(infcx);
    debug!(?value);
    let normalized_value = ocx.normalize(&cause, param_env, value);
    debug!(?normalized_value);
    debug!("select_all_or_error start");
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        return Err(errors);
    }
    debug!("select_all_or_error complete");
    let resolved_value = infcx.resolve_vars_if_possible(normalized_value);
    debug!(?resolved_value);
    Ok(resolved_value)
}

/// Process an obligation (and any nested obligations that come from it) to
/// completion, returning any errors
pub fn fully_solve_obligation<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: PredicateObligation<'tcx>,
) -> Vec<FulfillmentError<'tcx>> {
    fully_solve_obligations(infcx, [obligation])
}

/// Process a set of obligations (and any nested obligations that come from them)
/// to completion
pub fn fully_solve_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
) -> Vec<FulfillmentError<'tcx>> {
    let ocx = ObligationCtxt::new(infcx);
    ocx.register_obligations(obligations);
    ocx.select_all_or_error()
}

/// Process a bound (and any nested obligations that come from it) to completion.
/// This is a convenience function for traits that have no generic arguments, such
/// as auto traits, and builtin traits like Copy or Sized.
pub fn fully_solve_bound<'tcx>(
    infcx: &InferCtxt<'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    bound: DefId,
) -> Vec<FulfillmentError<'tcx>> {
    let tcx = infcx.tcx;
    let trait_ref = tcx.mk_trait_ref(bound, [ty]);
    let obligation = Obligation::new(tcx, cause, param_env, ty::Binder::dummy(trait_ref));

    fully_solve_obligation(infcx, obligation)
}

/// Normalizes the predicates and checks whether they hold in an empty environment. If this
/// returns true, then either normalize encountered an error or one of the predicates did not
/// hold. Used when creating vtables to check for unsatisfiable methods.
pub fn impossible_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: Vec<ty::Predicate<'tcx>>,
) -> bool {
    debug!("impossible_predicates(predicates={:?})", predicates);

    let infcx = tcx.infer_ctxt().build();
    let param_env = ty::ParamEnv::reveal_all();
    let ocx = ObligationCtxt::new(&infcx);
    let predicates = ocx.normalize(&ObligationCause::dummy(), param_env, predicates);
    for predicate in predicates {
        let obligation = Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate);
        ocx.register_obligation(obligation);
    }
    let errors = ocx.select_all_or_error();

    let result = !errors.is_empty();
    debug!("impossible_predicates = {:?}", result);
    result
}

fn subst_and_check_impossible_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (DefId, SubstsRef<'tcx>),
) -> bool {
    debug!("subst_and_check_impossible_predicates(key={:?})", key);

    let mut predicates = tcx.predicates_of(key.0).instantiate(tcx, key.1).predicates;

    // Specifically check trait fulfillment to avoid an error when trying to resolve
    // associated items.
    if let Some(trait_def_id) = tcx.trait_of_item(key.0) {
        let trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, key.1);
        predicates.push(ty::Binder::dummy(trait_ref).to_predicate(tcx));
    }

    predicates.retain(|predicate| !predicate.needs_subst());
    let result = impossible_predicates(tcx, predicates);

    debug!("subst_and_check_impossible_predicates(key={:?}) = {:?}", key, result);
    result
}

/// Checks whether a trait's method is impossible to call on a given impl.
///
/// This only considers predicates that reference the impl's generics, and not
/// those that reference the method's generics.
fn is_impossible_method(tcx: TyCtxt<'_>, (impl_def_id, trait_item_def_id): (DefId, DefId)) -> bool {
    struct ReferencesOnlyParentGenerics<'tcx> {
        tcx: TyCtxt<'tcx>,
        generics: &'tcx ty::Generics,
        trait_item_def_id: DefId,
    }
    impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for ReferencesOnlyParentGenerics<'tcx> {
        type BreakTy = ();
        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            // If this is a parameter from the trait item's own generics, then bail
            if let ty::Param(param) = t.kind()
                && let param_def_id = self.generics.type_param(param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            t.super_visit_with(self)
        }
        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ReEarlyBound(param) = r.kind()
                && let param_def_id = self.generics.region_param(&param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            r.super_visit_with(self)
        }
        fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ConstKind::Param(param) = ct.kind()
                && let param_def_id = self.generics.const_param(&param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            ct.super_visit_with(self)
        }
    }

    let generics = tcx.generics_of(trait_item_def_id);
    let predicates = tcx.predicates_of(trait_item_def_id);
    let impl_trait_ref = tcx
        .impl_trait_ref(impl_def_id)
        .expect("expected impl to correspond to trait")
        .subst_identity();
    let param_env = tcx.param_env(impl_def_id);

    let mut visitor = ReferencesOnlyParentGenerics { tcx, generics, trait_item_def_id };
    let predicates_for_trait = predicates.predicates.iter().filter_map(|(pred, span)| {
        pred.visit_with(&mut visitor).is_continue().then(|| {
            Obligation::new(
                tcx,
                ObligationCause::dummy_with_span(*span),
                param_env,
                ty::EarlyBinder(*pred).subst(tcx, impl_trait_ref.substs),
            )
        })
    });

    let infcx = tcx.infer_ctxt().ignoring_regions().build();
    for obligation in predicates_for_trait {
        // Ignore overflow error, to be conservative.
        if let Ok(result) = infcx.evaluate_obligation(&obligation)
            && !result.may_apply()
        {
            return true;
        }
    }
    false
}

pub fn provide(providers: &mut ty::query::Providers) {
    object_safety::provide(providers);
    vtable::provide(providers);
    *providers = ty::query::Providers {
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        subst_and_check_impossible_predicates,
        check_tys_might_be_eq: misc::check_tys_might_be_eq,
        is_impossible_method,
        ..*providers
    };
}
