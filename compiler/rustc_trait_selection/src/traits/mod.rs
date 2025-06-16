//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod auto_trait;
pub(crate) mod coherence;
pub mod const_evaluatable;
mod dyn_compatibility;
pub mod effects;
mod engine;
mod fulfill;
pub mod misc;
pub mod normalize;
pub mod outlives_bounds;
pub mod project;
pub mod query;
#[allow(hidden_glob_reexports)]
mod select;
mod specialize;
mod structural_normalize;
#[allow(hidden_glob_reexports)]
mod util;
pub mod vtable;
pub mod wf;

use std::fmt::Debug;
use std::ops::ControlFlow;

use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
pub use rustc_infer::traits::*;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{
    self, GenericArgs, GenericArgsRef, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypingMode, Upcast,
};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use tracing::{debug, instrument};

pub use self::coherence::{
    InCrate, IsFirstInputType, OrphanCheckErr, OrphanCheckMode, OverlapResult, UncoveredTyParams,
    add_placeholder_note, orphan_check_trait_ref, overlapping_impls,
};
pub use self::dyn_compatibility::{
    DynCompatibilityViolation, dyn_compatibility_violations_for_assoc_item,
    hir_ty_lowering_dyn_compatibility_violations, is_vtable_safe_method,
};
pub use self::engine::{ObligationCtxt, TraitEngineExt};
pub use self::fulfill::{FulfillmentContext, OldSolverError, PendingPredicateObligation};
pub use self::normalize::NormalizeExt;
pub use self::project::{normalize_inherent_projection, normalize_projection_term};
pub use self::select::{
    EvaluationCache, EvaluationResult, IntercrateAmbiguityCause, OverflowError, SelectionCache,
    SelectionContext,
};
pub use self::specialize::specialization_graph::{
    FutureCompatOverlapError, FutureCompatOverlapErrorKind,
};
pub use self::specialize::{
    OverlapError, specialization_graph, translate_args, translate_args_with_cause,
};
pub use self::structural_normalize::StructurallyNormalizeExt;
pub use self::util::{
    BoundVarReplacer, PlaceholderReplacer, elaborate, expand_trait_aliases, impl_item_is_final,
    sizedness_fast_path, supertrait_def_ids, supertraits, transitive_bounds_that_define_assoc_item,
    upcast_choices, with_replaced_escaping_bound_vars,
};
use crate::error_reporting::InferCtxtErrorExt;
use crate::infer::outlives::env::OutlivesEnvironment;
use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::regions::InferCtxtRegionExt;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;

#[derive(Debug)]
pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>,
    /// Diagnostics only: the 'root' obligation which resulted in
    /// the failure to process `obligation`. This is the obligation
    /// that was initially passed to `register_predicate_obligation`
    pub root_obligation: PredicateObligation<'tcx>,
}

impl<'tcx> FulfillmentError<'tcx> {
    pub fn new(
        obligation: PredicateObligation<'tcx>,
        code: FulfillmentErrorCode<'tcx>,
        root_obligation: PredicateObligation<'tcx>,
    ) -> FulfillmentError<'tcx> {
        FulfillmentError { obligation, code, root_obligation }
    }

    pub fn is_true_error(&self) -> bool {
        match self.code {
            FulfillmentErrorCode::Select(_)
            | FulfillmentErrorCode::Project(_)
            | FulfillmentErrorCode::Subtype(_, _)
            | FulfillmentErrorCode::ConstEquate(_, _) => true,
            FulfillmentErrorCode::Cycle(_) | FulfillmentErrorCode::Ambiguity { overflow: _ } => {
                false
            }
        }
    }
}

#[derive(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    /// Inherently impossible to fulfill; this trait is implemented if and only
    /// if it is already implemented.
    Cycle(PredicateObligations<'tcx>),
    Select(SelectionError<'tcx>),
    Project(MismatchedProjectionTypes<'tcx>),
    Subtype(ExpectedFound<Ty<'tcx>>, TypeError<'tcx>), // always comes from a SubtypePredicate
    ConstEquate(ExpectedFound<ty::Const<'tcx>>, TypeError<'tcx>),
    Ambiguity {
        /// Overflow is only `Some(suggest_recursion_limit)` when using the next generation
        /// trait solver `-Znext-solver`. With the old solver overflow is eagerly handled by
        /// emitting a fatal error instead.
        overflow: Option<bool>,
    },
}

impl<'tcx> Debug for FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FulfillmentErrorCode::Select(ref e) => write!(f, "{e:?}"),
            FulfillmentErrorCode::Project(ref e) => write!(f, "{e:?}"),
            FulfillmentErrorCode::Subtype(ref a, ref b) => {
                write!(f, "CodeSubtypeError({a:?}, {b:?})")
            }
            FulfillmentErrorCode::ConstEquate(ref a, ref b) => {
                write!(f, "CodeConstEquateError({a:?}, {b:?})")
            }
            FulfillmentErrorCode::Ambiguity { overflow: None } => write!(f, "Ambiguity"),
            FulfillmentErrorCode::Ambiguity { overflow: Some(suggest_increasing_limit) } => {
                write!(f, "Overflow({suggest_increasing_limit})")
            }
            FulfillmentErrorCode::Cycle(ref cycle) => write!(f, "Cycle({cycle:?})"),
        }
    }
}

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
    pred_known_to_hold_modulo_regions(infcx, param_env, trait_ref)
}

/// FIXME(@lcnr): this function doesn't seem right and shouldn't exist?
///
/// Ping me on zulip if you want to use this method and need help with finding
/// an appropriate replacement.
#[instrument(level = "debug", skip(infcx, param_env, pred), ret)]
fn pred_known_to_hold_modulo_regions<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pred: impl Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
) -> bool {
    let obligation = Obligation::new(infcx.tcx, ObligationCause::dummy(), param_env, pred);

    let result = infcx.evaluate_obligation_no_overflow(&obligation);
    debug!(?result);

    if result.must_apply_modulo_regions() {
        true
    } else if result.may_apply() && !infcx.next_trait_solver() {
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
                [] => infcx.resolve_vars_if_possible(goal) == goal,

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
    let infcx = tcx.infer_ctxt().ignoring_regions().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
    let predicates = ocx.normalize(&cause, elaborated_env, predicates);

    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let reported = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(reported);
    }

    debug!("do_normalize_predicates: normalized predicates = {:?}", predicates);

    // We can use the `elaborated_env` here; the region code only
    // cares about declarations like `'a: 'b`.
    // FIXME: It's very weird that we ignore region obligations but apparently
    // still need to use `resolve_regions` as we need the resolved regions in
    // the normalized predicates.
    let errors = infcx.resolve_regions(cause.body_id, elaborated_env, []);
    if !errors.is_empty() {
        tcx.dcx().span_delayed_bug(
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
            if tcx.features().generic_const_exprs() || tcx.next_trait_solver_globally() {
                return predicate;
            }

            struct ConstNormalizer<'tcx>(TyCtxt<'tcx>);

            impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ConstNormalizer<'tcx> {
                fn cx(&self) -> TyCtxt<'tcx> {
                    self.0
                }

                fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
                    // FIXME(return_type_notation): track binders in this normalizer, as
                    // `ty::Const::normalize` can only work with properly preserved binders.

                    if c.has_escaping_bound_vars() {
                        return ty::Const::new_misc_error(self.0);
                    }

                    // While it is pretty sus to be evaluating things with an empty param env, it
                    // should actually be okay since without `feature(generic_const_exprs)` the only
                    // const arguments that have a non-empty param env are array repeat counts. These
                    // do not appear in the type system though.
                    if let ty::ConstKind::Unevaluated(uv) = c.kind()
                        && self.0.def_kind(uv.def) == DefKind::AnonConst
                    {
                        let infcx = self.0.infer_ctxt().build(TypingMode::non_body_analysis());
                        let c = evaluate_const(&infcx, c, ty::ParamEnv::empty());
                        // We should never wind up with any `infcx` local state when normalizing anon consts
                        // under min const generics.
                        assert!(!c.has_infer() && !c.has_placeholders());
                        return c;
                    }

                    c
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
            predicate.fold_with(&mut ConstNormalizer(tcx))
        }),
    )
    .collect();

    debug!("normalize_param_env_or_error: elaborated-predicates={:?}", predicates);

    let elaborated_env = ty::ParamEnv::new(tcx.mk_clauses(&predicates));
    if !elaborated_env.has_aliases() {
        return elaborated_env;
    }

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
        .extract_if(.., |predicate| {
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
    let outlives_env = ty::ParamEnv::new(tcx.mk_clauses_from_iter(outlives_env));
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
    ty::ParamEnv::new(tcx.mk_clauses(&predicates))
}

#[derive(Debug)]
pub enum EvaluateConstErr {
    /// The constant being evaluated was either a generic parameter or inference variable, *or*,
    /// some unevaluated constant with either generic parameters or inference variables in its
    /// generic arguments.
    HasGenericsOrInfers,
    /// The type this constant evalauted to is not valid for use in const generics. This should
    /// always result in an error when checking the constant is correctly typed for the parameter
    /// it is an argument to, so a bug is delayed when encountering this.
    InvalidConstParamTy(ErrorGuaranteed),
    /// CTFE failed to evaluate the constant in some unrecoverable way (e.g. encountered a `panic!`).
    /// This is also used when the constant was already tainted by error.
    EvaluationFailure(ErrorGuaranteed),
}

// FIXME(BoxyUwU): Private this once we `generic_const_exprs` isn't doing its own normalization routine
// FIXME(generic_const_exprs): Consider accepting a `ty::UnevaluatedConst` when we are not rolling our own
// normalization scheme
/// Evaluates a type system constant returning a `ConstKind::Error` in cases where CTFE failed and
/// returning the passed in constant if it was not fully concrete (i.e. depended on generic parameters
/// or inference variables)
///
/// You should not call this function unless you are implementing normalization itself. Prefer to use
/// `normalize_erasing_regions` or the `normalize` functions on `ObligationCtxt`/`FnCtxt`/`InferCtxt`.
pub fn evaluate_const<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ct: ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> ty::Const<'tcx> {
    match try_evaluate_const(infcx, ct, param_env) {
        Ok(ct) => ct,
        Err(EvaluateConstErr::EvaluationFailure(e) | EvaluateConstErr::InvalidConstParamTy(e)) => {
            ty::Const::new_error(infcx.tcx, e)
        }
        Err(EvaluateConstErr::HasGenericsOrInfers) => ct,
    }
}

// FIXME(BoxyUwU): Private this once we `generic_const_exprs` isn't doing its own normalization routine
// FIXME(generic_const_exprs): Consider accepting a `ty::UnevaluatedConst` when we are not rolling our own
// normalization scheme
/// Evaluates a type system constant making sure to not allow constants that depend on generic parameters
/// or inference variables to succeed in evaluating.
///
/// You should not call this function unless you are implementing normalization itself. Prefer to use
/// `normalize_erasing_regions` or the `normalize` functions on `ObligationCtxt`/`FnCtxt`/`InferCtxt`.
#[instrument(level = "debug", skip(infcx), ret)]
pub fn try_evaluate_const<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ct: ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> Result<ty::Const<'tcx>, EvaluateConstErr> {
    let tcx = infcx.tcx;
    let ct = infcx.resolve_vars_if_possible(ct);
    debug!(?ct);

    match ct.kind() {
        ty::ConstKind::Value(..) => Ok(ct),
        ty::ConstKind::Error(e) => Err(EvaluateConstErr::EvaluationFailure(e)),
        ty::ConstKind::Param(_)
        | ty::ConstKind::Infer(_)
        | ty::ConstKind::Bound(_, _)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Expr(_) => Err(EvaluateConstErr::HasGenericsOrInfers),
        ty::ConstKind::Unevaluated(uv) => {
            let opt_anon_const_kind =
                (tcx.def_kind(uv.def) == DefKind::AnonConst).then(|| tcx.anon_const_kind(uv.def));

            // Postpone evaluation of constants that depend on generic parameters or
            // inference variables.
            //
            // We use `TypingMode::PostAnalysis` here which is not *technically* correct
            // to be revealing opaque types here as borrowcheck has not run yet. However,
            // CTFE itself uses `TypingMode::PostAnalysis` unconditionally even during
            // typeck and not doing so has a lot of (undesirable) fallout (#101478, #119821).
            // As a result we always use a revealed env when resolving the instance to evaluate.
            //
            // FIXME: `const_eval_resolve_for_typeck` should probably just modify the env itself
            // instead of having this logic here
            let (args, typing_env) = match opt_anon_const_kind {
                // We handle `generic_const_exprs` separately as reasonable ways of handling constants in the type system
                // completely fall apart under `generic_const_exprs` and makes this whole function Really hard to reason
                // about if you have to consider gce whatsoever.
                Some(ty::AnonConstKind::GCE) => {
                    if uv.has_non_region_infer() || uv.has_non_region_param() {
                        // `feature(generic_const_exprs)` causes anon consts to inherit all parent generics. This can cause
                        // inference variables and generic parameters to show up in `ty::Const` even though the anon const
                        // does not actually make use of them. We handle this case specially and attempt to evaluate anyway.
                        match tcx.thir_abstract_const(uv.def) {
                            Ok(Some(ct)) => {
                                let ct = tcx.expand_abstract_consts(ct.instantiate(tcx, uv.args));
                                if let Err(e) = ct.error_reported() {
                                    return Err(EvaluateConstErr::EvaluationFailure(e));
                                } else if ct.has_non_region_infer() || ct.has_non_region_param() {
                                    // If the anon const *does* actually use generic parameters or inference variables from
                                    // the generic arguments provided for it, then we should *not* attempt to evaluate it.
                                    return Err(EvaluateConstErr::HasGenericsOrInfers);
                                } else {
                                    let args =
                                        replace_param_and_infer_args_with_placeholder(tcx, uv.args);
                                    let typing_env = infcx
                                        .typing_env(tcx.erase_regions(param_env))
                                        .with_post_analysis_normalized(tcx);
                                    (args, typing_env)
                                }
                            }
                            Err(_) | Ok(None) => {
                                let args = GenericArgs::identity_for_item(tcx, uv.def);
                                let typing_env = ty::TypingEnv::post_analysis(tcx, uv.def);
                                (args, typing_env)
                            }
                        }
                    } else {
                        let typing_env = infcx
                            .typing_env(tcx.erase_regions(param_env))
                            .with_post_analysis_normalized(tcx);
                        (uv.args, typing_env)
                    }
                }
                Some(ty::AnonConstKind::RepeatExprCount) => {
                    if uv.has_non_region_infer() {
                        // Diagnostics will sometimes replace the identity args of anon consts in
                        // array repeat expr counts with inference variables so we have to handle this
                        // even though it is not something we should ever actually encounter.
                        //
                        // Array repeat expr counts are allowed to syntactically use generic parameters
                        // but must not actually depend on them in order to evalaute successfully. This means
                        // that it is actually fine to evalaute them in their own environment rather than with
                        // the actually provided generic arguments.
                        tcx.dcx().delayed_bug("AnonConst with infer args but no error reported");
                    }

                    // The generic args of repeat expr counts under `min_const_generics` are not supposed to
                    // affect evaluation of the constant as this would make it a "truly" generic const arg.
                    // To prevent this we discard all the generic arguments and evalaute with identity args
                    // and in its own environment instead of the current environment we are normalizing in.
                    let args = GenericArgs::identity_for_item(tcx, uv.def);
                    let typing_env = ty::TypingEnv::post_analysis(tcx, uv.def);

                    (args, typing_env)
                }
                _ => {
                    // We are only dealing with "truly" generic/uninferred constants here:
                    // - GCEConsts have been handled separately
                    // - Repeat expr count back compat consts have also been handled separately
                    // So we are free to simply defer evaluation here.
                    //
                    // FIXME: This assumes that `args` are normalized which is not necessarily true
                    //
                    // Const patterns are converted to type system constants before being
                    // evaluated. However, we don't care about them here as pattern evaluation
                    // logic does not go through type system normalization. If it did this would
                    // be a backwards compatibility problem as we do not enforce "syntactic" non-
                    // usage of generic parameters like we do here.
                    if uv.args.has_non_region_param() || uv.args.has_non_region_infer() {
                        return Err(EvaluateConstErr::HasGenericsOrInfers);
                    }

                    let typing_env = infcx
                        .typing_env(tcx.erase_regions(param_env))
                        .with_post_analysis_normalized(tcx);
                    (uv.args, typing_env)
                }
            };

            let uv = ty::UnevaluatedConst::new(uv.def, args);
            let erased_uv = tcx.erase_regions(uv);

            use rustc_middle::mir::interpret::ErrorHandled;
            // FIXME: `def_span` will point at the definition of this const; ideally, we'd point at
            // where it gets used as a const generic.
            match tcx.const_eval_resolve_for_typeck(typing_env, erased_uv, tcx.def_span(uv.def)) {
                Ok(Ok(val)) => Ok(ty::Const::new_value(
                    tcx,
                    val,
                    tcx.type_of(uv.def).instantiate(tcx, uv.args),
                )),
                Ok(Err(_)) => {
                    let e = tcx.dcx().delayed_bug(
                        "Type system constant with non valtree'able type evaluated but no error emitted",
                    );
                    Err(EvaluateConstErr::InvalidConstParamTy(e))
                }
                Err(ErrorHandled::Reported(info, _)) => {
                    Err(EvaluateConstErr::EvaluationFailure(info.into()))
                }
                Err(ErrorHandled::TooGeneric(_)) => Err(EvaluateConstErr::HasGenericsOrInfers),
            }
        }
    }
}

/// Replaces args that reference param or infer variables with suitable
/// placeholders. This function is meant to remove these param and infer
/// args when they're not actually needed to evaluate a constant.
fn replace_param_and_infer_args_with_placeholder<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: GenericArgsRef<'tcx>,
) -> GenericArgsRef<'tcx> {
    struct ReplaceParamAndInferWithPlaceholder<'tcx> {
        tcx: TyCtxt<'tcx>,
        idx: ty::BoundVar,
    }

    impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceParamAndInferWithPlaceholder<'tcx> {
        fn cx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
            if let ty::Infer(_) = t.kind() {
                let idx = self.idx;
                self.idx += 1;
                Ty::new_placeholder(
                    self.tcx,
                    ty::PlaceholderType {
                        universe: ty::UniverseIndex::ROOT,
                        bound: ty::BoundTy { var: idx, kind: ty::BoundTyKind::Anon },
                    },
                )
            } else {
                t.super_fold_with(self)
            }
        }

        fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
            if let ty::ConstKind::Infer(_) = c.kind() {
                let idx = self.idx;
                self.idx += 1;
                ty::Const::new_placeholder(
                    self.tcx,
                    ty::PlaceholderConst { universe: ty::UniverseIndex::ROOT, bound: idx },
                )
            } else {
                c.super_fold_with(self)
            }
        }
    }

    args.fold_with(&mut ReplaceParamAndInferWithPlaceholder { tcx, idx: ty::BoundVar::ZERO })
}

/// Normalizes the predicates and checks whether they hold in an empty environment. If this
/// returns true, then either normalize encountered an error or one of the predicates did not
/// hold. Used when creating vtables to check for unsatisfiable methods. This should not be
/// used during analysis.
pub fn impossible_predicates<'tcx>(tcx: TyCtxt<'tcx>, predicates: Vec<ty::Clause<'tcx>>) -> bool {
    debug!("impossible_predicates(predicates={:?})", predicates);
    let (infcx, param_env) = tcx
        .infer_ctxt()
        .with_next_trait_solver(true)
        .build_with_typing_env(ty::TypingEnv::fully_monomorphized());

    let ocx = ObligationCtxt::new(&infcx);
    let predicates = ocx.normalize(&ObligationCause::dummy(), param_env, predicates);
    for predicate in predicates {
        let obligation = Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate);
        ocx.register_obligation(obligation);
    }

    // Use `select_where_possible` to only return impossible for true errors,
    // and not ambiguities or overflows. Since the new trait solver forces
    // some currently undetected overlap between `dyn Trait: Trait` built-in
    // vs user-written impls to AMBIGUOUS, this may return ambiguity even
    // with no infer vars. There may also be ways to encounter ambiguity due
    // to post-mono overflow.
    let true_errors = ocx.select_where_possible();
    if !true_errors.is_empty() {
        return true;
    }

    false
}

fn instantiate_and_check_impossible_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (DefId, GenericArgsRef<'tcx>),
) -> bool {
    debug!("instantiate_and_check_impossible_predicates(key={:?})", key);

    let mut predicates = tcx.predicates_of(key.0).instantiate(tcx, key.1).predicates;

    // Specifically check trait fulfillment to avoid an error when trying to resolve
    // associated items.
    if let Some(trait_def_id) = tcx.trait_of_item(key.0) {
        let trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, key.1);
        predicates.push(trait_ref.upcast(tcx));
    }

    predicates.retain(|predicate| !predicate.has_param());
    let result = impossible_predicates(tcx, predicates);

    debug!("instantiate_and_check_impossible_predicates(key={:?}) = {:?}", key, result);
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
        type Result = ControlFlow<()>;
        fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
            // If this is a parameter from the trait item's own generics, then bail
            if let ty::Param(param) = *t.kind()
                && let param_def_id = self.generics.type_param(param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            t.super_visit_with(self)
        }
        fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
            if let ty::ReEarlyParam(param) = r.kind()
                && let param_def_id = self.generics.region_param(param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        }
        fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
            if let ty::ConstKind::Param(param) = ct.kind()
                && let param_def_id = self.generics.const_param(param, self.tcx).def_id
                && self.tcx.parent(param_def_id) == self.trait_item_def_id
            {
                return ControlFlow::Break(());
            }
            ct.super_visit_with(self)
        }
    }

    let generics = tcx.generics_of(trait_item_def_id);
    let predicates = tcx.predicates_of(trait_item_def_id);

    // Be conservative in cases where we have `W<T: ?Sized>` and a method like `Self: Sized`,
    // since that method *may* have some substitutions where the predicates hold.
    //
    // This replicates the logic we use in coherence.
    let infcx = tcx
        .infer_ctxt()
        .ignoring_regions()
        .with_next_trait_solver(true)
        .build(TypingMode::Coherence);
    let param_env = ty::ParamEnv::empty();
    let fresh_args = infcx.fresh_args_for_item(tcx.def_span(impl_def_id), impl_def_id);

    let impl_trait_ref = tcx
        .impl_trait_ref(impl_def_id)
        .expect("expected impl to correspond to trait")
        .instantiate(tcx, fresh_args);

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

    let ocx = ObligationCtxt::new(&infcx);
    ocx.register_obligations(predicates_for_trait);
    !ocx.select_where_possible().is_empty()
}

pub fn provide(providers: &mut Providers) {
    dyn_compatibility::provide(providers);
    vtable::provide(providers);
    *providers = Providers {
        specialization_graph_of: specialize::specialization_graph_provider,
        specializes: specialize::specializes,
        specialization_enabled_in: specialize::specialization_enabled_in,
        instantiate_and_check_impossible_predicates,
        is_impossible_associated_item,
        ..*providers
    };
}
