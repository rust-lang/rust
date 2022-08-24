//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod auto_trait;
mod chalk_fulfill;
pub mod codegen;
mod coherence;
pub mod const_evaluatable;
mod engine;
pub mod error_reporting;
mod fulfill;
pub mod misc;
mod object_safety;
mod on_unimplemented;
pub mod outlives_bounds;
mod project;
pub mod query;
pub(crate) mod relationships;
mod select;
mod specialize;
mod structural_match;
mod util;
pub mod wf;

use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::traits::error_reporting::InferCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_infer::traits::TraitEngineExt as _;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{InternalSubsts, SubstsRef};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{
    self, DefIdTree, GenericParamDefKind, Subst, ToPredicate, Ty, TyCtxt, TypeSuperVisitable,
    VtblEntry,
};
use rustc_span::{sym, Span};
use smallvec::SmallVec;

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
pub use self::on_unimplemented::{OnUnimplementedDirective, OnUnimplementedNote};
pub use self::project::{normalize, normalize_projection_type, normalize_to};
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
    /// Canonicalized queries get dummy spans and hence
    /// must generally propagate errors to
    /// pre-canonicalization callsites.
    Canonical,
}

/// Creates predicate obligations from the generic bounds.
pub fn predicates_for_generics<'tcx>(
    cause: impl Fn(usize, Span) -> ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    generic_bounds: ty::InstantiatedPredicates<'tcx>,
) -> impl Iterator<Item = PredicateObligation<'tcx>> {
    let generic_bounds = generic_bounds;
    debug!("predicates_for_generics(generic_bounds={:?})", generic_bounds);

    std::iter::zip(generic_bounds.predicates, generic_bounds.spans).enumerate().map(
        move |(idx, (predicate, span))| Obligation {
            cause: cause(idx, span),
            recursion_depth: 0,
            param_env: param_env,
            predicate,
        },
    )
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

    let trait_ref =
        ty::Binder::dummy(ty::TraitRef { def_id, substs: infcx.tcx.mk_substs_trait(ty, &[]) });
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

        // We can use a dummy node-id here because we won't pay any mind
        // to region obligations that arise (there shouldn't really be any
        // anyhow).
        let cause = ObligationCause::misc(span, hir::CRATE_HIR_ID);

        // The handling of regions in this area of the code is terrible,
        // see issue #29149. We should be able to improve on this with
        // NLL.
        let errors = fully_solve_bound(infcx, cause, param_env, ty, def_id);

        // Note: we only assume something is `Copy` if we can
        // *definitively* show that it implements `Copy`. Otherwise,
        // assume it is move; linear is always ok.
        match &errors[..] {
            [] => {
                debug!(
                    "type_known_to_meet_bound_modulo_regions: ty={:?} bound={} success",
                    ty,
                    infcx.tcx.def_path_str(def_id)
                );
                true
            }
            errors => {
                debug!(
                    ?ty,
                    bound = %infcx.tcx.def_path_str(def_id),
                    ?errors,
                    "type_known_to_meet_bound_modulo_regions"
                );
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
    tcx.infer_ctxt().ignoring_regions().enter(|infcx| {
        let predicates = match fully_normalize(&infcx, cause, elaborated_env, predicates) {
            Ok(predicates) => predicates,
            Err(errors) => {
                let reported = infcx.report_fulfillment_errors(&errors, None, false);
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
                format!(
                    "failed region resolution while normalizing {elaborated_env:?}: {errors:?}"
                ),
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
    })
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
        tcx.intern_predicates(&predicates),
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
    let outlives_env: Vec<_> =
        non_outlives_predicates.iter().chain(&outlives_predicates).cloned().collect();
    let outlives_env = ty::ParamEnv::new(
        tcx.intern_predicates(&outlives_env),
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
        tcx.intern_predicates(&predicates),
        unnormalized_env.reveal(),
        unnormalized_env.constness(),
    )
}

/// Normalize a type and process all resulting obligations, returning any errors
pub fn fully_normalize<'a, 'tcx, T>(
    infcx: &InferCtxt<'a, 'tcx>,
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

    let mut fulfill_cx = FulfillmentContext::new();
    for obligation in obligations {
        fulfill_cx.register_predicate_obligation(infcx, obligation);
    }

    debug!("fully_normalize: select_all_or_error start");
    let errors = fulfill_cx.select_all_or_error(infcx);
    if !errors.is_empty() {
        return Err(errors);
    }
    debug!("fully_normalize: select_all_or_error complete");
    let resolved_value = infcx.resolve_vars_if_possible(normalized_value);
    debug!("fully_normalize: resolved_value={:?}", resolved_value);
    Ok(resolved_value)
}

/// Process an obligation (and any nested obligations that come from it) to
/// completion, returning any errors
pub fn fully_solve_obligation<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    obligation: PredicateObligation<'tcx>,
) -> Vec<FulfillmentError<'tcx>> {
    let mut engine = <dyn TraitEngine<'tcx>>::new(infcx.tcx);
    engine.register_predicate_obligation(infcx, obligation);
    engine.select_all_or_error(infcx)
}

/// Process a set of obligations (and any nested obligations that come from them)
/// to completion
pub fn fully_solve_obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
) -> Vec<FulfillmentError<'tcx>> {
    let mut engine = <dyn TraitEngine<'tcx>>::new(infcx.tcx);
    engine.register_predicate_obligations(infcx, obligations);
    engine.select_all_or_error(infcx)
}

/// Process a bound (and any nested obligations that come from it) to completion.
/// This is a convenience function for traits that have no generic arguments, such
/// as auto traits, and builtin traits like Copy or Sized.
pub fn fully_solve_bound<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    bound: DefId,
) -> Vec<FulfillmentError<'tcx>> {
    let mut engine = <dyn TraitEngine<'tcx>>::new(infcx.tcx);
    engine.register_bound(infcx, param_env, ty, bound, cause);
    engine.select_all_or_error(infcx)
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
        let ocx = ObligationCtxt::new(&infcx);
        let predicates = ocx.normalize(ObligationCause::dummy(), param_env, predicates);
        for predicate in predicates {
            let obligation = Obligation::new(ObligationCause::dummy(), param_env, predicate);
            ocx.register_obligation(obligation);
        }
        let errors = ocx.select_all_or_error();

        // Clean up after ourselves
        let _ = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();

        !errors.is_empty()
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

    // Specifically check trait fulfillment to avoid an error when trying to resolve
    // associated items.
    if let Some(trait_def_id) = tcx.trait_of_item(key.0) {
        let trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, key.1);
        predicates.push(ty::Binder::dummy(trait_ref).to_poly_trait_predicate().to_predicate(tcx));
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
fn is_impossible_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    (impl_def_id, trait_item_def_id): (DefId, DefId),
) -> bool {
    struct ReferencesOnlyParentGenerics<'tcx> {
        tcx: TyCtxt<'tcx>,
        generics: &'tcx ty::Generics,
        trait_item_def_id: DefId,
    }
    impl<'tcx> ty::TypeVisitor<'tcx> for ReferencesOnlyParentGenerics<'tcx> {
        type BreakTy = ();
        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            // If this is a parameter from the trait item's own generics, then bail
            if let ty::Param(param) = t.kind()
                && let param_def_id = self.generics.type_param(param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::BREAK;
            }
            t.super_visit_with(self)
        }
        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ReEarlyBound(param) = r.kind()
                && let param_def_id = self.generics.region_param(&param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::BREAK;
            }
            r.super_visit_with(self)
        }
        fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::ConstKind::Param(param) = ct.kind()
                && let param_def_id = self.generics.const_param(&param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::BREAK;
            }
            ct.super_visit_with(self)
        }
    }

    let generics = tcx.generics_of(trait_item_def_id);
    let predicates = tcx.predicates_of(trait_item_def_id);
    let impl_trait_ref =
        tcx.impl_trait_ref(impl_def_id).expect("expected impl to correspond to trait");
    let param_env = tcx.param_env(impl_def_id);

    let mut visitor = ReferencesOnlyParentGenerics { tcx, generics, trait_item_def_id };
    let predicates_for_trait = predicates.predicates.iter().filter_map(|(pred, span)| {
        if pred.visit_with(&mut visitor).is_continue() {
            Some(Obligation::new(
                ObligationCause::dummy_with_span(*span),
                param_env,
                ty::EarlyBinder(*pred).subst(tcx, impl_trait_ref.substs),
            ))
        } else {
            None
        }
    });

    tcx.infer_ctxt().ignoring_regions().enter(|ref infcx| {
        let mut fulfill_ctxt = <dyn TraitEngine<'_>>::new(tcx);
        fulfill_ctxt.register_predicate_obligations(infcx, predicates_for_trait);
        !fulfill_ctxt.select_all_or_error(infcx).is_empty()
    })
}

#[derive(Clone, Debug)]
enum VtblSegment<'tcx> {
    MetadataDSA,
    TraitOwnEntries { trait_ref: ty::PolyTraitRef<'tcx>, emit_vptr: bool },
}

/// Prepare the segments for a vtable
fn prepare_vtable_segments<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
    mut segment_visitor: impl FnMut(VtblSegment<'tcx>) -> ControlFlow<T>,
) -> Option<T> {
    // The following constraints holds for the final arrangement.
    // 1. The whole virtual table of the first direct super trait is included as the
    //    the prefix. If this trait doesn't have any super traits, then this step
    //    consists of the dsa metadata.
    // 2. Then comes the proper pointer metadata(vptr) and all own methods for all
    //    other super traits except those already included as part of the first
    //    direct super trait virtual table.
    // 3. finally, the own methods of this trait.

    // This has the advantage that trait upcasting to the first direct super trait on each level
    // is zero cost, and to another trait includes only replacing the pointer with one level indirection,
    // while not using too much extra memory.

    // For a single inheritance relationship like this,
    //   D --> C --> B --> A
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, C, D

    // For a multiple inheritance relationship like this,
    //   D --> C --> A
    //           \-> B
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, B-vptr, C, D

    // For a diamond inheritance relationship like this,
    //   D --> B --> A
    //     \-> C -/
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, C, C-vptr, D

    // For a more complex inheritance relationship like this:
    //   O --> G --> C --> A
    //     \     \     \-> B
    //     |     |-> F --> D
    //     |           \-> E
    //     |-> N --> J --> H
    //           \     \-> I
    //           |-> M --> K
    //                 \-> L
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, B-vptr, C, D, D-vptr, E, E-vptr, F, F-vptr, G,
    //  H, H-vptr, I, I-vptr, J, J-vptr, K, K-vptr, L, L-vptr, M, M-vptr,
    //  N, N-vptr, O

    // emit dsa segment first.
    if let ControlFlow::Break(v) = (segment_visitor)(VtblSegment::MetadataDSA) {
        return Some(v);
    }

    let mut emit_vptr_on_new_entry = false;
    let mut visited = util::PredicateSet::new(tcx);
    let predicate = trait_ref.without_const().to_predicate(tcx);
    let mut stack: SmallVec<[(ty::PolyTraitRef<'tcx>, _, _); 5]> =
        smallvec![(trait_ref, emit_vptr_on_new_entry, None)];
    visited.insert(predicate);

    // the main traversal loop:
    // basically we want to cut the inheritance directed graph into a few non-overlapping slices of nodes
    // that each node is emitted after all its descendents have been emitted.
    // so we convert the directed graph into a tree by skipping all previously visited nodes using a visited set.
    // this is done on the fly.
    // Each loop run emits a slice - it starts by find a "childless" unvisited node, backtracking upwards, and it
    // stops after it finds a node that has a next-sibling node.
    // This next-sibling node will used as the starting point of next slice.

    // Example:
    // For a diamond inheritance relationship like this,
    //   D#1 --> B#0 --> A#0
    //     \-> C#1 -/

    // Starting point 0 stack [D]
    // Loop run #0: Stack after diving in is [D B A], A is "childless"
    // after this point, all newly visited nodes won't have a vtable that equals to a prefix of this one.
    // Loop run #0: Emitting the slice [B A] (in reverse order), B has a next-sibling node, so this slice stops here.
    // Loop run #0: Stack after exiting out is [D C], C is the next starting point.
    // Loop run #1: Stack after diving in is [D C], C is "childless", since its child A is skipped(already emitted).
    // Loop run #1: Emitting the slice [D C] (in reverse order). No one has a next-sibling node.
    // Loop run #1: Stack after exiting out is []. Now the function exits.

    loop {
        // dive deeper into the stack, recording the path
        'diving_in: loop {
            if let Some((inner_most_trait_ref, _, _)) = stack.last() {
                let inner_most_trait_ref = *inner_most_trait_ref;
                let mut direct_super_traits_iter = tcx
                    .super_predicates_of(inner_most_trait_ref.def_id())
                    .predicates
                    .into_iter()
                    .filter_map(move |(pred, _)| {
                        pred.subst_supertrait(tcx, &inner_most_trait_ref).to_opt_poly_trait_pred()
                    });

                'diving_in_skip_visited_traits: loop {
                    if let Some(next_super_trait) = direct_super_traits_iter.next() {
                        if visited.insert(next_super_trait.to_predicate(tcx)) {
                            // We're throwing away potential constness of super traits here.
                            // FIXME: handle ~const super traits
                            let next_super_trait = next_super_trait.map_bound(|t| t.trait_ref);
                            stack.push((
                                next_super_trait,
                                emit_vptr_on_new_entry,
                                Some(direct_super_traits_iter),
                            ));
                            break 'diving_in_skip_visited_traits;
                        } else {
                            continue 'diving_in_skip_visited_traits;
                        }
                    } else {
                        break 'diving_in;
                    }
                }
            }
        }

        // Other than the left-most path, vptr should be emitted for each trait.
        emit_vptr_on_new_entry = true;

        // emit innermost item, move to next sibling and stop there if possible, otherwise jump to outer level.
        'exiting_out: loop {
            if let Some((inner_most_trait_ref, emit_vptr, siblings_opt)) = stack.last_mut() {
                if let ControlFlow::Break(v) = (segment_visitor)(VtblSegment::TraitOwnEntries {
                    trait_ref: *inner_most_trait_ref,
                    emit_vptr: *emit_vptr,
                }) {
                    return Some(v);
                }

                'exiting_out_skip_visited_traits: loop {
                    if let Some(siblings) = siblings_opt {
                        if let Some(next_inner_most_trait_ref) = siblings.next() {
                            if visited.insert(next_inner_most_trait_ref.to_predicate(tcx)) {
                                // We're throwing away potential constness of super traits here.
                                // FIXME: handle ~const super traits
                                let next_inner_most_trait_ref =
                                    next_inner_most_trait_ref.map_bound(|t| t.trait_ref);
                                *inner_most_trait_ref = next_inner_most_trait_ref;
                                *emit_vptr = emit_vptr_on_new_entry;
                                break 'exiting_out;
                            } else {
                                continue 'exiting_out_skip_visited_traits;
                            }
                        }
                    }
                    stack.pop();
                    continue 'exiting_out;
                }
            }
            // all done
            return None;
        }
    }
}

fn dump_vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    trait_ref: ty::PolyTraitRef<'tcx>,
    entries: &[VtblEntry<'tcx>],
) {
    let msg = format!("vtable entries for `{}`: {:#?}", trait_ref, entries);
    tcx.sess.struct_span_err(sp, &msg).emit();
}

fn own_existential_vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyExistentialTraitRef<'tcx>,
) -> &'tcx [DefId] {
    let trait_methods = tcx
        .associated_items(trait_ref.def_id())
        .in_definition_order()
        .filter(|item| item.kind == ty::AssocKind::Fn);
    // Now list each method's DefId (for within its trait).
    let own_entries = trait_methods.filter_map(move |trait_method| {
        debug!("own_existential_vtable_entry: trait_method={:?}", trait_method);
        let def_id = trait_method.def_id;

        // Some methods cannot be called on an object; skip those.
        if !is_vtable_safe_method(tcx, trait_ref.def_id(), &trait_method) {
            debug!("own_existential_vtable_entry: not vtable safe");
            return None;
        }

        Some(def_id)
    });

    tcx.arena.alloc_from_iter(own_entries.into_iter())
}

/// Given a trait `trait_ref`, iterates the vtable entries
/// that come from `trait_ref`, including its supertraits.
fn vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> &'tcx [VtblEntry<'tcx>] {
    debug!("vtable_entries({:?})", trait_ref);

    let mut entries = vec![];

    let vtable_segment_callback = |segment| -> ControlFlow<()> {
        match segment {
            VtblSegment::MetadataDSA => {
                entries.extend(TyCtxt::COMMON_VTABLE_ENTRIES);
            }
            VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                let existential_trait_ref = trait_ref
                    .map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));

                // Lookup the shape of vtable for the trait.
                let own_existential_entries =
                    tcx.own_existential_vtable_entries(existential_trait_ref);

                let own_entries = own_existential_entries.iter().copied().map(|def_id| {
                    debug!("vtable_entries: trait_method={:?}", def_id);

                    // The method may have some early-bound lifetimes; add regions for those.
                    let substs = trait_ref.map_bound(|trait_ref| {
                        InternalSubsts::for_item(tcx, def_id, |param, _| match param.kind {
                            GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                            GenericParamDefKind::Type { .. }
                            | GenericParamDefKind::Const { .. } => {
                                trait_ref.substs[param.index as usize]
                            }
                        })
                    });

                    // The trait type may have higher-ranked lifetimes in it;
                    // erase them if they appear, so that we get the type
                    // at some particular call site.
                    let substs = tcx
                        .normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), substs);

                    // It's possible that the method relies on where-clauses that
                    // do not hold for this particular set of type parameters.
                    // Note that this method could then never be called, so we
                    // do not want to try and codegen it, in that case (see #23435).
                    let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, substs);
                    if impossible_predicates(tcx, predicates.predicates) {
                        debug!("vtable_entries: predicates do not hold");
                        return VtblEntry::Vacant;
                    }

                    let instance = ty::Instance::resolve_for_vtable(
                        tcx,
                        ty::ParamEnv::reveal_all(),
                        def_id,
                        substs,
                    )
                    .expect("resolution failed during building vtable representation");
                    VtblEntry::Method(instance)
                });

                entries.extend(own_entries);

                if emit_vptr {
                    entries.push(VtblEntry::TraitVPtr(trait_ref));
                }
            }
        }

        ControlFlow::Continue(())
    };

    let _ = prepare_vtable_segments(tcx, trait_ref, vtable_segment_callback);

    if tcx.has_attr(trait_ref.def_id(), sym::rustc_dump_vtable) {
        let sp = tcx.def_span(trait_ref.def_id());
        dump_vtable_entries(tcx, sp, trait_ref, &entries);
    }

    tcx.arena.alloc_from_iter(entries.into_iter())
}

/// Find slot base for trait methods within vtable entries of another trait
fn vtable_trait_first_method_offset<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        ty::PolyTraitRef<'tcx>, // trait_to_be_found
        ty::PolyTraitRef<'tcx>, // trait_owning_vtable
    ),
) -> usize {
    let (trait_to_be_found, trait_owning_vtable) = key;

    // #90177
    let trait_to_be_found_erased = tcx.erase_regions(trait_to_be_found);

    let vtable_segment_callback = {
        let mut vtable_base = 0;

        move |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    vtable_base += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                    if tcx.erase_regions(trait_ref) == trait_to_be_found_erased {
                        return ControlFlow::Break(vtable_base);
                    }
                    vtable_base += util::count_own_vtable_entries(tcx, trait_ref);
                    if emit_vptr {
                        vtable_base += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        }
    };

    if let Some(vtable_base) =
        prepare_vtable_segments(tcx, trait_owning_vtable, vtable_segment_callback)
    {
        vtable_base
    } else {
        bug!("Failed to find info for expected trait in vtable");
    }
}

/// Find slot offset for trait vptr within vtable entries of another trait
pub fn vtable_trait_upcasting_coercion_new_vptr_slot<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        Ty<'tcx>, // trait object type whose trait owning vtable
        Ty<'tcx>, // trait object for supertrait
    ),
) -> Option<usize> {
    let (source, target) = key;
    assert!(matches!(&source.kind(), &ty::Dynamic(..)) && !source.needs_infer());
    assert!(matches!(&target.kind(), &ty::Dynamic(..)) && !target.needs_infer());

    // this has been typecked-before, so diagnostics is not really needed.
    let unsize_trait_did = tcx.require_lang_item(LangItem::Unsize, None);

    let trait_ref = ty::TraitRef {
        def_id: unsize_trait_did,
        substs: tcx.mk_substs_trait(source, &[target.into()]),
    };
    let obligation = Obligation::new(
        ObligationCause::dummy(),
        ty::ParamEnv::reveal_all(),
        ty::Binder::dummy(ty::TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Positive,
        }),
    );

    let implsrc = tcx.infer_ctxt().enter(|infcx| {
        let mut selcx = SelectionContext::new(&infcx);
        selcx.select(&obligation).unwrap()
    });

    let Some(ImplSource::TraitUpcasting(implsrc_traitcasting)) = implsrc else {
        bug!();
    };

    implsrc_traitcasting.vtable_vptr_slot
}

pub fn provide(providers: &mut ty::query::Providers) {
    object_safety::provide(providers);
    structural_match::provide(providers);
    *providers = ty::query::Providers {
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        codegen_fulfill_obligation: codegen::codegen_fulfill_obligation,
        own_existential_vtable_entries,
        vtable_entries,
        vtable_trait_upcasting_coercion_new_vptr_slot,
        subst_and_check_impossible_predicates,
        is_impossible_method,
        try_unify_abstract_consts: |tcx, param_env_and| {
            let (param_env, (a, b)) = param_env_and.into_parts();
            const_evaluatable::try_unify_abstract_consts(tcx, (a, b), param_env)
        },
        ..*providers
    };
}
