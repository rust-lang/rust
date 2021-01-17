//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod auto_trait;
mod chalk_fulfill;
pub mod codegen;
mod coherence;
mod const_evaluatable;
mod engine;
pub mod error_reporting;
mod fulfill;
pub mod misc;
mod object_safety;
mod on_unimplemented;
mod project;
pub mod query;
mod select;
mod specialize;
mod structural_match;
mod util;
pub mod wf;

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{InferCtxt, RegionckMode, TyCtxtInferExt};
use crate::traits::error_reporting::InferCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{InternalSubsts, SubstsRef};
use rustc_middle::ty::{
    self, GenericParamDefKind, ParamEnv, ToPredicate, Ty, TyCtxt, WithConstness,
};
use rustc_span::Span;

use std::fmt::Debug;

pub use self::FulfillmentErrorCode::*;
pub use self::ImplSource::*;
pub use self::ObligationCauseCode::*;
pub use self::SelectionError::*;

pub use self::coherence::{add_placeholder_note, orphan_check, overlapping_impls};
pub use self::coherence::{OrphanCheckErr, OverlapResult};
pub use self::engine::TraitEngineExt;
pub use self::fulfill::{FulfillmentContext, PendingPredicateObligation};
pub use self::object_safety::astconv_object_safety_violations;
pub use self::object_safety::is_vtable_safe_method;
pub use self::object_safety::MethodViolationCode;
pub use self::object_safety::ObjectSafetyViolation;
pub use self::on_unimplemented::{OnUnimplementedDirective, OnUnimplementedNote};
pub use self::project::{normalize, normalize_projection_type, normalize_to};
pub use self::select::{EvaluationCache, SelectionCache, SelectionContext};
pub use self::select::{EvaluationResult, IntercrateAmbiguityCause, OverflowError};
pub use self::specialize::specialization_graph::FutureCompatOverlapError;
pub use self::specialize::specialization_graph::FutureCompatOverlapErrorKind;
pub use self::specialize::{specialization_graph, translate_substs, OverlapError};
pub use self::structural_match::search_for_structural_match_violation;
pub use self::structural_match::NonStructuralMatchTy;
pub use self::util::{elaborate_predicates, elaborate_trait_ref, elaborate_trait_refs};
pub use self::util::{expand_trait_aliases, TraitAliasExpander};
pub use self::util::{
    get_vtable_index_of_object_method, impl_item_is_final, predicate_for_trait_def, upcast_choices,
};
pub use self::util::{
    supertrait_def_ids, supertraits, transitive_bounds, SupertraitDefIds, Supertraits,
};

pub use self::chalk_fulfill::FulfillmentContext as ChalkFulfillmentContext;

pub use rustc_infer::traits::*;

/// Whether to skip the leak check, as part of a future compatibility warning step.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SkipLeakCheck {
    Yes,
    No,
}

impl SkipLeakCheck {
    fn is_yes(self) -> bool {
        self == SkipLeakCheck::Yes
    }
}

/// The "default" for skip-leak-check corresponds to the current
/// behavior (do not skip the leak check) -- not the behavior we are
/// transitioning into.
impl Default for SkipLeakCheck {
    fn default() -> Self {
        SkipLeakCheck::No
    }
}

/// The mode that trait queries run in.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TraitQueryMode {
    /// Standard/un-canonicalized queries get accurate
    /// spans etc. passed in and hence can do reasonable
    /// error reporting on their own.
    Standard,
    /// Canonicalized queries get dummy spans and hence
    /// must generally propagate errors to
    /// pre-canonicalization callsites.
    Canonical,
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    generic_bounds: ty::InstantiatedPredicates<'tcx>,
) -> impl Iterator<Item = PredicateObligation<'tcx>> {
    util::predicates_for_generics(cause, 0, param_env, generic_bounds)
}

/// Determines whether the type `ty` is known to meet `bound` and
/// returns true if so. Returns false if `ty` either does not meet
/// `bound` or is not known to meet bound (note that this is
/// conservative towards *no impl*, which is the opposite of the
/// `evaluate` methods).
pub fn type_known_to_meet_bound_modulo_regions<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    def_id: DefId,
    span: Span,
) -> bool {
    debug!(
        "type_known_to_meet_bound_modulo_regions(ty={:?}, bound={:?})",
        ty,
        infcx.tcx.def_path_str(def_id)
    );

    let trait_ref = ty::TraitRef { def_id, substs: infcx.tcx.mk_substs_trait(ty, &[]) };
    let obligation = Obligation {
        param_env,
        cause: ObligationCause::misc(span, hir::CRATE_HIR_ID),
        recursion_depth: 0,
        predicate: trait_ref.without_const().to_predicate(infcx.tcx),
    };

    let result = infcx.predicate_must_hold_modulo_regions(&obligation);
    debug!(
        "type_known_to_meet_ty={:?} bound={} => {:?}",
        ty,
        infcx.tcx.def_path_str(def_id),
        result
    );

    if result && ty.has_infer_types_or_consts() {
        // Because of inference "guessing", selection can sometimes claim
        // to succeed while the success requires a guess. To ensure
        // this function's result remains infallible, we must confirm
        // that guess. While imperfect, I believe this is sound.

        // The handling of regions in this area of the code is terrible,
        // see issue #29149. We should be able to improve on this with
        // NLL.
        let mut fulfill_cx = FulfillmentContext::new_ignoring_regions();

        // We can use a dummy node-id here because we won't pay any mind
        // to region obligations that arise (there shouldn't really be any
        // anyhow).
        let cause = ObligationCause::misc(span, hir::CRATE_HIR_ID);

        fulfill_cx.register_bound(infcx, param_env, ty, def_id, cause);

        // Note: we only assume something is `Copy` if we can
        // *definitively* show that it implements `Copy`. Otherwise,
        // assume it is move; linear is always ok.
        match fulfill_cx.select_all_or_error(infcx) {
            Ok(()) => {
                debug!(
                    "type_known_to_meet_bound_modulo_regions: ty={:?} bound={} success",
                    ty,
                    infcx.tcx.def_path_str(def_id)
                );
                true
            }
            Err(e) => {
                debug!(
                    "type_known_to_meet_bound_modulo_regions: ty={:?} bound={} errors={:?}",
                    ty,
                    infcx.tcx.def_path_str(def_id),
                    e
                );
                false
            }
        }
    } else {
        result
    }
}

fn do_normalize_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    region_context: DefId,
    cause: ObligationCause<'tcx>,
    elaborated_env: ty::ParamEnv<'tcx>,
    predicates: Vec<ty::Predicate<'tcx>>,
) -> Result<Vec<ty::Predicate<'tcx>>, ErrorReported> {
    debug!(
        "do_normalize_predicates(predicates={:?}, region_context={:?}, cause={:?})",
        predicates, region_context, cause,
    );
    let span = cause.span;
    tcx.infer_ctxt().enter(|infcx| {
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
        let fulfill_cx = FulfillmentContext::new_ignoring_regions();
        let predicates =
            match fully_normalize(&infcx, fulfill_cx, cause, elaborated_env, predicates) {
                Ok(predicates) => predicates,
                Err(errors) => {
                    infcx.report_fulfillment_errors(&errors, None, false);
                    return Err(ErrorReported);
                }
            };

        debug!("do_normalize_predictes: normalized predicates = {:?}", predicates);

        // We can use the `elaborated_env` here; the region code only
        // cares about declarations like `'a: 'b`.
        let outlives_env = OutlivesEnvironment::new(elaborated_env);

        infcx.resolve_regions_and_report_errors(
            region_context,
            &outlives_env,
            RegionckMode::default(),
        );

        let predicates = match infcx.fully_resolve(predicates) {
            Ok(predicates) => predicates,
            Err(fixup_err) => {
                // If we encounter a fixup error, it means that some type
                // variable wound up unconstrained. I actually don't know
                // if this can happen, and I certainly don't expect it to
                // happen often, but if it did happen it probably
                // represents a legitimate failure due to some kind of
                // unconstrained variable, and it seems better not to ICE,
                // all things considered.
                tcx.sess.span_err(span, &fixup_err.to_string());
                return Err(ErrorReported);
            }
        };
        if predicates.needs_infer() {
            tcx.sess.delay_span_bug(span, "encountered inference variables after `fully_resolve`");
            Err(ErrorReported)
        } else {
            Ok(predicates)
        }
    })
}

// FIXME: this is gonna need to be removed ...
/// Normalizes the parameter environment, reporting errors if they occur.
pub fn normalize_param_env_or_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    region_context: DefId,
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
    // and errors will get reported then; so after typeck we
    // can be sure that no errors should occur.

    debug!(
        "normalize_param_env_or_error(region_context={:?}, unnormalized_env={:?}, cause={:?})",
        region_context, unnormalized_env, cause
    );

    let mut predicates: Vec<_> =
        util::elaborate_predicates(tcx, unnormalized_env.caller_bounds().into_iter())
            .map(|obligation| obligation.predicate)
            .collect();

    debug!("normalize_param_env_or_error: elaborated-predicates={:?}", predicates);

    let elaborated_env =
        ty::ParamEnv::new(tcx.intern_predicates(&predicates), unnormalized_env.reveal());

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
    // This works fairly well because trait matching  does not actually care about param-env
    // TypeOutlives predicates - these are normally used by regionck.
    let outlives_predicates: Vec<_> = predicates
        .drain_filter(|predicate| {
            matches!(predicate.kind().skip_binder(), ty::PredicateKind::TypeOutlives(..))
        })
        .collect();

    debug!(
        "normalize_param_env_or_error: predicates=(non-outlives={:?}, outlives={:?})",
        predicates, outlives_predicates
    );
    let non_outlives_predicates = match do_normalize_predicates(
        tcx,
        region_context,
        cause.clone(),
        elaborated_env,
        predicates,
    ) {
        Ok(predicates) => predicates,
        // An unnormalized env is better than nothing.
        Err(ErrorReported) => {
            debug!("normalize_param_env_or_error: errored resolving non-outlives predicates");
            return elaborated_env;
        }
    };

    debug!("normalize_param_env_or_error: non-outlives predicates={:?}", non_outlives_predicates);

    // Not sure whether it is better to include the unnormalized TypeOutlives predicates
    // here. I believe they should not matter, because we are ignoring TypeOutlives param-env
    // predicates here anyway. Keeping them here anyway because it seems safer.
    let outlives_env: Vec<_> =
        non_outlives_predicates.iter().chain(&outlives_predicates).cloned().collect();
    let outlives_env =
        ty::ParamEnv::new(tcx.intern_predicates(&outlives_env), unnormalized_env.reveal());
    let outlives_predicates = match do_normalize_predicates(
        tcx,
        region_context,
        cause,
        outlives_env,
        outlives_predicates,
    ) {
        Ok(predicates) => predicates,
        // An unnormalized env is better than nothing.
        Err(ErrorReported) => {
            debug!("normalize_param_env_or_error: errored resolving outlives predicates");
            return elaborated_env;
        }
    };
    debug!("normalize_param_env_or_error: outlives predicates={:?}", outlives_predicates);

    let mut predicates = non_outlives_predicates;
    predicates.extend(outlives_predicates);
    debug!("normalize_param_env_or_error: final predicates={:?}", predicates);
    ty::ParamEnv::new(tcx.intern_predicates(&predicates), unnormalized_env.reveal())
}

pub fn fully_normalize<'a, 'tcx, T>(
    infcx: &InferCtxt<'a, 'tcx>,
    mut fulfill_cx: FulfillmentContext<'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: T,
) -> Result<T, Vec<FulfillmentError<'tcx>>>
where
    T: TypeFoldable<'tcx>,
{
    debug!("fully_normalize_with_fulfillcx(value={:?})", value);
    let selcx = &mut SelectionContext::new(infcx);
    let Normalized { value: normalized_value, obligations } =
        project::normalize(selcx, param_env, cause, value);
    debug!(
        "fully_normalize: normalized_value={:?} obligations={:?}",
        normalized_value, obligations
    );
    for obligation in obligations {
        fulfill_cx.register_predicate_obligation(selcx.infcx(), obligation);
    }

    debug!("fully_normalize: select_all_or_error start");
    fulfill_cx.select_all_or_error(infcx)?;
    debug!("fully_normalize: select_all_or_error complete");
    let resolved_value = infcx.resolve_vars_if_possible(normalized_value);
    debug!("fully_normalize: resolved_value={:?}", resolved_value);
    Ok(resolved_value)
}

/// Normalizes the predicates and checks whether they hold in an empty environment. If this
/// returns true, then either normalize encountered an error or one of the predicates did not
/// hold. Used when creating vtables to check for unsatisfiable methods.
pub fn impossible_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: Vec<ty::Predicate<'tcx>>,
) -> bool {
    debug!("impossible_predicates(predicates={:?})", predicates);

    let result = tcx.infer_ctxt().enter(|infcx| {
        let param_env = ty::ParamEnv::reveal_all();
        let mut selcx = SelectionContext::new(&infcx);
        let mut fulfill_cx = FulfillmentContext::new();
        let cause = ObligationCause::dummy();
        let Normalized { value: predicates, obligations } =
            normalize(&mut selcx, param_env, cause.clone(), predicates);
        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(&infcx, obligation);
        }
        for predicate in predicates {
            let obligation = Obligation::new(cause.clone(), param_env, predicate);
            fulfill_cx.register_predicate_obligation(&infcx, obligation);
        }

        fulfill_cx.select_all_or_error(&infcx).is_err()
    });
    debug!("impossible_predicates = {:?}", result);
    result
}

fn subst_and_check_impossible_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (DefId, SubstsRef<'tcx>),
) -> bool {
    debug!("subst_and_check_impossible_predicates(key={:?})", key);

    let mut predicates = tcx.predicates_of(key.0).instantiate(tcx, key.1).predicates;
    predicates.retain(|predicate| !predicate.needs_subst());
    let result = impossible_predicates(tcx, predicates);

    debug!("subst_and_check_impossible_predicates(key={:?}) = {:?}", key, result);
    result
}

/// Given a trait `trait_ref`, iterates the vtable entries
/// that come from `trait_ref`, including its supertraits.
#[inline] // FIXME(#35870): avoid closures being unexported due to `impl Trait`.
fn vtable_methods<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> &'tcx [Option<(DefId, SubstsRef<'tcx>)>] {
    debug!("vtable_methods({:?})", trait_ref);

    tcx.arena.alloc_from_iter(supertraits(tcx, trait_ref).flat_map(move |trait_ref| {
        let trait_methods = tcx
            .associated_items(trait_ref.def_id())
            .in_definition_order()
            .filter(|item| item.kind == ty::AssocKind::Fn);

        // Now list each method's DefId and InternalSubsts (for within its trait).
        // If the method can never be called from this object, produce None.
        trait_methods.map(move |trait_method| {
            debug!("vtable_methods: trait_method={:?}", trait_method);
            let def_id = trait_method.def_id;

            // Some methods cannot be called on an object; skip those.
            if !is_vtable_safe_method(tcx, trait_ref.def_id(), &trait_method) {
                debug!("vtable_methods: not vtable safe");
                return None;
            }

            // The method may have some early-bound lifetimes; add regions for those.
            let substs = trait_ref.map_bound(|trait_ref| {
                InternalSubsts::for_item(tcx, def_id, |param, _| match param.kind {
                    GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const => {
                        trait_ref.substs[param.index as usize]
                    }
                })
            });

            // The trait type may have higher-ranked lifetimes in it;
            // erase them if they appear, so that we get the type
            // at some particular call site.
            let substs =
                tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), substs);

            // It's possible that the method relies on where-clauses that
            // do not hold for this particular set of type parameters.
            // Note that this method could then never be called, so we
            // do not want to try and codegen it, in that case (see #23435).
            let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, substs);
            if impossible_predicates(tcx, predicates.predicates) {
                debug!("vtable_methods: predicates do not hold");
                return None;
            }

            Some((def_id, substs))
        })
    }))
}

/// Check whether a `ty` implements given trait(trait_def_id).
///
/// NOTE: Always return `false` for a type which needs inference.
fn type_implements_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        DefId,    // trait_def_id,
        Ty<'tcx>, // type
        SubstsRef<'tcx>,
        ParamEnv<'tcx>,
    ),
) -> bool {
    let (trait_def_id, ty, params, param_env) = key;

    debug!(
        "type_implements_trait: trait_def_id={:?}, type={:?}, params={:?}, param_env={:?}",
        trait_def_id, ty, params, param_env
    );

    let trait_ref = ty::TraitRef { def_id: trait_def_id, substs: tcx.mk_substs_trait(ty, params) };

    let obligation = Obligation {
        cause: ObligationCause::dummy(),
        param_env,
        recursion_depth: 0,
        predicate: trait_ref.without_const().to_predicate(tcx),
    };
    tcx.infer_ctxt().enter(|infcx| infcx.predicate_must_hold_modulo_regions(&obligation))
}

pub fn provide(providers: &mut ty::query::Providers) {
    object_safety::provide(providers);
    structural_match::provide(providers);
    *providers = ty::query::Providers {
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        codegen_fulfill_obligation: codegen::codegen_fulfill_obligation,
        vtable_methods,
        type_implements_trait,
        subst_and_check_impossible_predicates,
        mir_abstract_const: |tcx, def_id| {
            let def_id = def_id.expect_local();
            if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
                tcx.mir_abstract_const_of_const_arg(def)
            } else {
                const_evaluatable::mir_abstract_const(tcx, ty::WithOptConstParam::unknown(def_id))
            }
        },
        mir_abstract_const_of_const_arg: |tcx, (did, param_did)| {
            const_evaluatable::mir_abstract_const(
                tcx,
                ty::WithOptConstParam { did, const_param_did: Some(param_did) },
            )
        },
        try_unify_abstract_consts: const_evaluatable::try_unify_abstract_consts,
        ..*providers
    };
}
