//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod auto_trait;
pub(crate) mod coherence;
pub mod const_evaluatable;
mod engine;
pub mod error_reporting;
mod fulfill;
pub mod misc;
mod object_safety;
pub mod outlives_bounds;
pub mod project;
pub mod query;
#[allow(hidden_glob_reexports)]
mod select;
mod specialize;
mod structural_match;
mod structural_normalize;
#[allow(hidden_glob_reexports)]
mod util;
pub mod vtable;
pub mod wf;

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::traits::error_reporting::TypeErrCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_errors::ErrorGuaranteed;
use rustc_middle::query::Providers;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::visit::{TypeVisitable, TypeVisitableExt};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt, TypeFolder, TypeSuperVisitable};
use rustc_middle::ty::{GenericArgs, GenericArgsRef};
use rustc_span::def_id::DefId;
use rustc_span::Span;

use std::fmt::Debug;
use std::ops::ControlFlow;

pub(crate) use self::project::{needs_normalization, BoundVarReplacer, PlaceholderReplacer};

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
pub use self::project::NormalizeExt;
pub use self::project::{normalize_inherent_projection, normalize_projection_type};
pub use self::select::{EvaluationCache, SelectionCache, SelectionContext};
pub use self::select::{EvaluationResult, IntercrateAmbiguityCause, OverflowError};
pub use self::specialize::specialization_graph::FutureCompatOverlapError;
pub use self::specialize::specialization_graph::FutureCompatOverlapErrorKind;
pub use self::specialize::{
    specialization_graph, translate_args, translate_args_with_cause, OverlapError,
};
pub use self::structural_match::search_for_structural_match_violation;
pub use self::structural_normalize::StructurallyNormalizeExt;
pub use self::util::elaborate;
pub use self::util::{
    check_args_compatible, supertrait_def_ids, supertraits, transitive_bounds,
    transitive_bounds_that_define_assoc_item, SupertraitDefIds,
};
pub use self::util::{expand_trait_aliases, TraitAliasExpander};
pub use self::util::{get_vtable_index_of_object_method, impl_item_is_final, upcast_choices};

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
    generic_bounds.into_iter().enumerate().map(move |(idx, (clause, span))| Obligation {
        cause: cause(idx, span),
        recursion_depth: 0,
        param_env,
        predicate: clause.as_predicate(),
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
) -> bool {
    let trait_ref = ty::TraitRef::new(infcx.tcx, def_id, [ty]);
    pred_known_to_hold_modulo_regions(infcx, param_env, trait_ref.without_const())
}

/// FIXME(@lcnr): this function doesn't seem right and shouldn't exist?
///
/// Ping me on zulip if you want to use this method and need help with finding
/// an appropriate replacement.
#[instrument(level = "debug", skip(infcx, param_env, pred), ret)]
fn pred_known_to_hold_modulo_regions<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pred: impl ToPredicate<'tcx>,
) -> bool {
    let obligation = Obligation::new(infcx.tcx, ObligationCause::dummy(), param_env, pred);

    let result = infcx.evaluate_obligation_no_overflow(&obligation);
    debug!(?result);

    if result.must_apply_modulo_regions() {
        true
    } else if result.may_apply() {
        // Sometimes obligations are ambiguous because the recursive evaluator
        // is not smart enough, so we fall back to fulfillment when we're not certain
        // that an obligation holds or not. Even still, we must make sure that
        // the we do no inference in the process of checking this obligation.
        let goal = infcx.resolve_vars_if_possible((obligation.predicate, obligation.param_env));
        infcx.probe(|_| {
            let ocx = ObligationCtxt::new(infcx);
            ocx.register_obligation(obligation);

            let errors = ocx.select_all_or_error();
            match errors.as_slice() {
                // Only known to hold if we did no inference.
                [] => infcx.shallow_resolve(goal) == goal,

                errors => {
                    debug!(?errors);
                    false
                }
            }
        })
    } else {
        false
    }
}

#[instrument(level = "debug", skip(tcx, elaborated_env))]
fn do_normalize_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: ObligationCause<'tcx>,
    elaborated_env: ty::ParamEnv<'tcx>,
    predicates: Vec<ty::Clause<'tcx>>,
) -> Result<Vec<ty::Clause<'tcx>>, ErrorGuaranteed> {
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
            let reported = infcx.err_ctxt().report_fulfillment_errors(&errors);
            return Err(reported);
        }
    };

    debug!("do_normalize_predicates: normalized predicates = {:?}", predicates);

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
    let mut predicates: Vec<_> = util::elaborate(
        tcx,
        unnormalized_env.caller_bounds().into_iter().map(|predicate| {
            if tcx.features().generic_const_exprs {
                return predicate;
            }

            struct ConstNormalizer<'tcx>(TyCtxt<'tcx>);

            impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ConstNormalizer<'tcx> {
                fn interner(&self) -> TyCtxt<'tcx> {
                    self.0
                }

                fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
                    // While it is pretty sus to be evaluating things with an empty param env, it
                    // should actually be okay since without `feature(generic_const_exprs)` the only
                    // const arguments that have a non-empty param env are array repeat counts. These
                    // do not appear in the type system though.
                    c.eval(self.0, ty::ParamEnv::empty())
                }
            }

            // This whole normalization step is a hack to work around the fact that
            // `normalize_param_env_or_error` is fundamentally broken from using an
            // unnormalized param env with a trait solver that expects the param env
            // to be normalized.
            //
            // When normalizing the param env we can end up evaluating obligations
            // that have been normalized but can only be proven via a where clause
            // which is still in its unnormalized form. example:
            //
            // Attempting to prove `T: Trait<<u8 as Identity>::Assoc>` in a param env
            // with a `T: Trait<<u8 as Identity>::Assoc>` where clause will fail because
            // we first normalize obligations before proving them so we end up proving
            // `T: Trait<u8>`. Since lazy normalization is not implemented equating `u8`
            // with `<u8 as Identity>::Assoc` fails outright so we incorrectly believe that
            // we cannot prove `T: Trait<u8>`.
            //
            // The same thing is true for const generics- attempting to prove
            // `T: Trait<ConstKind::Unevaluated(...)>` with the same thing as a where clauses
            // will fail. After normalization we may be attempting to prove `T: Trait<4>` with
            // the unnormalized where clause `T: Trait<ConstKind::Unevaluated(...)>`. In order
            // for the obligation to hold `4` must be equal to `ConstKind::Unevaluated(...)`
            // but as we do not have lazy norm implemented, equating the two consts fails outright.
            //
            // Ideally we would not normalize consts here at all but it is required for backwards
            // compatibility. Eventually when lazy norm is implemented this can just be removed.
            // We do not normalize types here as there is no backwards compatibility requirement
            // for us to do so.
            //
            // FIXME(-Ztrait-solver=next): remove this hack since we have deferred projection equality
            predicate.fold_with(&mut ConstNormalizer(tcx))
        }),
    )
    .collect();

    debug!("normalize_param_env_or_error: elaborated-predicates={:?}", predicates);

    let elaborated_env = ty::ParamEnv::new(
        tcx.mk_clauses(&predicates),
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
        .extract_if(|predicate| {
            matches!(predicate.kind().skip_binder(), ty::ClauseKind::TypeOutlives(..))
        })
        .collect();

    debug!(
        "normalize_param_env_or_error: predicates=(non-outlives={:?}, outlives={:?})",
        predicates, outlives_predicates
    );
    let Ok(non_outlives_predicates) =
        do_normalize_predicates(tcx, cause.clone(), elaborated_env, predicates)
    else {
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
        tcx.mk_clauses_from_iter(outlives_env),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    );
    let Ok(outlives_predicates) =
        do_normalize_predicates(tcx, cause, outlives_env, outlives_predicates)
    else {
        // An unnormalized env is better than nothing.
        debug!("normalize_param_env_or_error: errored resolving outlives predicates");
        return elaborated_env;
    };
    debug!("normalize_param_env_or_error: outlives predicates={:?}", outlives_predicates);

    let mut predicates = non_outlives_predicates;
    predicates.extend(outlives_predicates);
    debug!("normalize_param_env_or_error: final predicates={:?}", predicates);
    ty::ParamEnv::new(
        tcx.mk_clauses(&predicates),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    )
}

/// Normalize a type and process all resulting obligations, returning any errors.
///
/// FIXME(-Ztrait-solver=next): This should be replaced by `At::deeply_normalize`
/// which has the same behavior with the new solver. Because using a separate
/// fulfillment context worsens caching in the old solver, `At::deeply_normalize`
/// is still lazy with the old solver as it otherwise negatively impacts perf.
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

/// Normalizes the predicates and checks whether they hold in an empty environment. If this
/// returns true, then either normalize encountered an error or one of the predicates did not
/// hold. Used when creating vtables to check for unsatisfiable methods.
pub fn impossible_predicates<'tcx>(tcx: TyCtxt<'tcx>, predicates: Vec<ty::Clause<'tcx>>) -> bool {
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
    key: (DefId, GenericArgsRef<'tcx>),
) -> bool {
    debug!("subst_and_check_impossible_predicates(key={:?})", key);

    let mut predicates = tcx.predicates_of(key.0).instantiate(tcx, key.1).predicates;

    // Specifically check trait fulfillment to avoid an error when trying to resolve
    // associated items.
    if let Some(trait_def_id) = tcx.trait_of_item(key.0) {
        let trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, key.1);
        predicates.push(ty::Binder::dummy(trait_ref).to_predicate(tcx));
    }

    predicates.retain(|predicate| !predicate.has_param());
    let result = impossible_predicates(tcx, predicates);

    debug!("subst_and_check_impossible_predicates(key={:?}) = {:?}", key, result);
    result
}

/// Checks whether a trait's associated item is impossible to reference on a given impl.
///
/// This only considers predicates that reference the impl's generics, and not
/// those that reference the method's generics.
fn is_impossible_associated_item(
    tcx: TyCtxt<'_>,
    (impl_def_id, trait_item_def_id): (DefId, DefId),
) -> bool {
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
            ControlFlow::Continue(())
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
        .instantiate_identity();
    let param_env = tcx.param_env(impl_def_id);

    let mut visitor = ReferencesOnlyParentGenerics { tcx, generics, trait_item_def_id };
    let predicates_for_trait = predicates.predicates.iter().filter_map(|(pred, span)| {
        pred.visit_with(&mut visitor).is_continue().then(|| {
            Obligation::new(
                tcx,
                ObligationCause::dummy_with_span(*span),
                param_env,
                ty::EarlyBinder::bind(*pred).instantiate(tcx, impl_trait_ref.args),
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

pub fn provide(providers: &mut Providers) {
    object_safety::provide(providers);
    vtable::provide(providers);
    *providers = Providers {
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        subst_and_check_impossible_predicates,
        check_tys_might_be_eq: misc::check_tys_might_be_eq,
        is_impossible_associated_item,
        ..*providers
    };
}
