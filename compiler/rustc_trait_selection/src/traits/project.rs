//! Code for projecting associated types out of trait references.

use std::ops::ControlFlow;

use rustc_data_structures::sso::SsoHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::infer::{DefineOpaqueTypes, RegionVariableOrigin};
use rustc_infer::traits::{ObligationCauseCode, PredicateObligations};
use rustc_middle::traits::select::OverflowError;
use rustc_middle::traits::{BuiltinImplSource, ImplSource, ImplSourceUserDefinedData};
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Term, Ty, TyCtxt, TypingMode, Upcast};
use rustc_middle::{bug, span_bug};
use rustc_span::sym;
use thin_vec::thin_vec;
use tracing::{debug, instrument};

use super::{
    MismatchedProjectionTypes, Normalized, NormalizedTerm, Obligation, ObligationCause,
    PredicateObligation, ProjectionCacheEntry, ProjectionCacheKey, Selection, SelectionContext,
    SelectionError, specialization_graph, translate_args, util,
};
use crate::errors::InherentProjectionNormalizationOverflow;
use crate::infer::{BoundRegionConversionTime, InferOk};
use crate::traits::normalize::{normalize_with_depth, normalize_with_depth_to};
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::select::ProjectionMatchesProjection;

pub type PolyProjectionObligation<'tcx> = Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> = Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTermObligation<'tcx> = Obligation<'tcx, ty::AliasTerm<'tcx>>;

pub(super) struct InProgress;

/// When attempting to resolve `<T as TraitRef>::Name` ...
#[derive(Debug)]
pub enum ProjectionError<'tcx> {
    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[derive(PartialEq, Eq, Debug)]
enum ProjectionCandidate<'tcx> {
    /// From a where-clause in the env or object type
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),

    /// From the definition of `Trait` when you have something like
    /// `<<A as Trait>::B as Trait2>::C`.
    TraitDef(ty::PolyProjectionPredicate<'tcx>),

    /// Bounds specified on an object type
    Object(ty::PolyProjectionPredicate<'tcx>),

    /// Built-in bound for a dyn async fn in trait
    ObjectRpitit,

    /// From an "impl" (or a "pseudo-impl" returned by select)
    Select(Selection<'tcx>),
}

enum ProjectionCandidateSet<'tcx> {
    None,
    Single(ProjectionCandidate<'tcx>),
    Ambiguous,
    Error(SelectionError<'tcx>),
}

impl<'tcx> ProjectionCandidateSet<'tcx> {
    fn mark_ambiguous(&mut self) {
        *self = ProjectionCandidateSet::Ambiguous;
    }

    fn mark_error(&mut self, err: SelectionError<'tcx>) {
        *self = ProjectionCandidateSet::Error(err);
    }

    // Returns true if the push was successful, or false if the candidate
    // was discarded -- this could be because of ambiguity, or because
    // a higher-priority candidate is already there.
    fn push_candidate(&mut self, candidate: ProjectionCandidate<'tcx>) -> bool {
        use self::ProjectionCandidate::*;
        use self::ProjectionCandidateSet::*;

        // This wacky variable is just used to try and
        // make code readable and avoid confusing paths.
        // It is assigned a "value" of `()` only on those
        // paths in which we wish to convert `*self` to
        // ambiguous (and return false, because the candidate
        // was not used). On other paths, it is not assigned,
        // and hence if those paths *could* reach the code that
        // comes after the match, this fn would not compile.
        let convert_to_ambiguous;

        match self {
            None => {
                *self = Single(candidate);
                return true;
            }

            Single(current) => {
                // Duplicates can happen inside ParamEnv. In the case, we
                // perform a lazy deduplication.
                if current == &candidate {
                    return false;
                }

                // Prefer where-clauses. As in select, if there are multiple
                // candidates, we prefer where-clause candidates over impls. This
                // may seem a bit surprising, since impls are the source of
                // "truth" in some sense, but in fact some of the impls that SEEM
                // applicable are not, because of nested obligations. Where
                // clauses are the safer choice. See the comment on
                // `select::SelectionCandidate` and #21974 for more details.
                match (current, candidate) {
                    (ParamEnv(..), ParamEnv(..)) => convert_to_ambiguous = (),
                    (ParamEnv(..), _) => return false,
                    (_, ParamEnv(..)) => bug!(
                        "should never prefer non-param-env candidates over param-env candidates"
                    ),
                    (_, _) => convert_to_ambiguous = (),
                }
            }

            Ambiguous | Error(..) => {
                return false;
            }
        }

        // We only ever get here when we moved from a single candidate
        // to ambiguous.
        let () = convert_to_ambiguous;
        *self = Ambiguous;
        false
    }
}

/// States returned from `poly_project_and_unify_type`. Takes the place
/// of the old return type, which was:
/// ```ignore (not-rust)
/// Result<
///     Result<Option<PredicateObligations<'tcx>>, InProgress>,
///     MismatchedProjectionTypes<'tcx>,
/// >
/// ```
pub(super) enum ProjectAndUnifyResult<'tcx> {
    /// The projection bound holds subject to the given obligations. If the
    /// projection cannot be normalized because the required trait bound does
    /// not hold, this is returned, with `obligations` being a predicate that
    /// cannot be proven.
    Holds(PredicateObligations<'tcx>),
    /// The projection cannot be normalized due to ambiguity. Resolving some
    /// inference variables in the projection may fix this.
    FailedNormalization,
    /// The project cannot be normalized because `poly_project_and_unify_type`
    /// is called recursively while normalizing the same projection.
    Recursive,
    // the projection can be normalized, but is not equal to the expected type.
    // Returns the type error that arose from the mismatch.
    MismatchedProjectionTypes(MismatchedProjectionTypes<'tcx>),
}

/// Evaluates constraints of the form:
/// ```ignore (not-rust)
/// for<...> <T as Trait>::U == V
/// ```
/// If successful, this may result in additional obligations. Also returns
/// the projection cache key used to track these additional obligations.
#[instrument(level = "debug", skip(selcx))]
pub(super) fn poly_project_and_unify_term<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &PolyProjectionObligation<'tcx>,
) -> ProjectAndUnifyResult<'tcx> {
    let infcx = selcx.infcx;
    let r = infcx.commit_if_ok(|_snapshot| {
        let placeholder_predicate = infcx.enter_forall_and_leak_universe(obligation.predicate);

        let placeholder_obligation = obligation.with(infcx.tcx, placeholder_predicate);
        match project_and_unify_term(selcx, &placeholder_obligation) {
            ProjectAndUnifyResult::MismatchedProjectionTypes(e) => Err(e),
            other => Ok(other),
        }
    });

    match r {
        Ok(inner) => inner,
        Err(err) => ProjectAndUnifyResult::MismatchedProjectionTypes(err),
    }
}

/// Evaluates constraints of the form:
/// ```ignore (not-rust)
/// <T as Trait>::U == V
/// ```
/// If successful, this may result in additional obligations.
///
/// See [poly_project_and_unify_term] for an explanation of the return value.
#[instrument(level = "debug", skip(selcx))]
fn project_and_unify_term<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionObligation<'tcx>,
) -> ProjectAndUnifyResult<'tcx> {
    let mut obligations = PredicateObligations::new();

    let infcx = selcx.infcx;
    let normalized = match opt_normalize_projection_term(
        selcx,
        obligation.param_env,
        obligation.predicate.projection_term,
        obligation.cause.clone(),
        obligation.recursion_depth,
        &mut obligations,
    ) {
        Ok(Some(n)) => n,
        Ok(None) => return ProjectAndUnifyResult::FailedNormalization,
        Err(InProgress) => return ProjectAndUnifyResult::Recursive,
    };
    debug!(?normalized, ?obligations, "project_and_unify_type result");
    let actual = obligation.predicate.term;
    // For an example where this is necessary see tests/ui/impl-trait/nested-return-type2.rs
    // This allows users to omit re-mentioning all bounds on an associated type and just use an
    // `impl Trait` for the assoc type to add more bounds.
    let InferOk { value: actual, obligations: new } =
        selcx.infcx.replace_opaque_types_with_inference_vars(
            actual,
            obligation.cause.body_id,
            obligation.cause.span,
            obligation.param_env,
        );
    obligations.extend(new);

    // Need to define opaque types to support nested opaque types like `impl Fn() -> impl Trait`
    match infcx.at(&obligation.cause, obligation.param_env).eq(
        DefineOpaqueTypes::Yes,
        normalized,
        actual,
    ) {
        Ok(InferOk { obligations: inferred_obligations, value: () }) => {
            obligations.extend(inferred_obligations);
            ProjectAndUnifyResult::Holds(obligations)
        }
        Err(err) => {
            debug!("equating types encountered error {:?}", err);
            ProjectAndUnifyResult::MismatchedProjectionTypes(MismatchedProjectionTypes { err })
        }
    }
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). If ambiguity arises, which implies that
/// there are unresolved type variables in the projection, we will
/// instantiate it with a fresh type variable `$X` and generate a new
/// obligation `<T as Trait>::Item == $X` for later.
pub fn normalize_projection_ty<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut PredicateObligations<'tcx>,
) -> Term<'tcx> {
    opt_normalize_projection_term(
        selcx,
        param_env,
        projection_ty.into(),
        cause.clone(),
        depth,
        obligations,
    )
    .ok()
    .flatten()
    .unwrap_or_else(move || {
        // if we bottom out in ambiguity, create a type variable
        // and a deferred predicate to resolve this when more type
        // information is available.

        selcx
            .infcx
            .projection_ty_to_infer(param_env, projection_ty, cause, depth + 1, obligations)
            .into()
    })
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). Returns `None` in the case of ambiguity,
/// which indicates that there are unbound type variables.
///
/// This function used to return `Option<NormalizedTy<'tcx>>`, which contains a
/// `Ty<'tcx>` and an obligations vector. But that obligation vector was very
/// often immediately appended to another obligations vector. So now this
/// function takes an obligations vector and appends to it directly, which is
/// slightly uglier but avoids the need for an extra short-lived allocation.
#[instrument(level = "debug", skip(selcx, param_env, cause, obligations))]
pub(super) fn opt_normalize_projection_term<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_term: ty::AliasTerm<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut PredicateObligations<'tcx>,
) -> Result<Option<Term<'tcx>>, InProgress> {
    let infcx = selcx.infcx;
    debug_assert!(!selcx.infcx.next_trait_solver());
    let projection_term = infcx.resolve_vars_if_possible(projection_term);
    let cache_key = ProjectionCacheKey::new(projection_term, param_env);

    // FIXME(#20304) For now, I am caching here, which is good, but it
    // means we don't capture the type variables that are created in
    // the case of ambiguity. Which means we may create a large stream
    // of such variables. OTOH, if we move the caching up a level, we
    // would not benefit from caching when proving `T: Trait<U=Foo>`
    // bounds. It might be the case that we want two distinct caches,
    // or else another kind of cache entry.
    let cache_entry = infcx.inner.borrow_mut().projection_cache().try_start(cache_key);
    match cache_entry {
        Ok(()) => debug!("no cache"),
        Err(ProjectionCacheEntry::Ambiguous) => {
            // If we found ambiguity the last time, that means we will continue
            // to do so until some type in the key changes (and we know it
            // hasn't, because we just fully resolved it).
            debug!("found cache entry: ambiguous");
            return Ok(None);
        }
        Err(ProjectionCacheEntry::InProgress) => {
            // Under lazy normalization, this can arise when
            // bootstrapping. That is, imagine an environment with a
            // where-clause like `A::B == u32`. Now, if we are asked
            // to normalize `A::B`, we will want to check the
            // where-clauses in scope. So we will try to unify `A::B`
            // with `A::B`, which can trigger a recursive
            // normalization.

            debug!("found cache entry: in-progress");

            // Cache that normalizing this projection resulted in a cycle. This
            // should ensure that, unless this happens within a snapshot that's
            // rolled back, fulfillment or evaluation will notice the cycle.
            infcx.inner.borrow_mut().projection_cache().recur(cache_key);
            return Err(InProgress);
        }
        Err(ProjectionCacheEntry::Recur) => {
            debug!("recur cache");
            return Err(InProgress);
        }
        Err(ProjectionCacheEntry::NormalizedTerm { ty, complete: _ }) => {
            // This is the hottest path in this function.
            //
            // If we find the value in the cache, then return it along
            // with the obligations that went along with it. Note
            // that, when using a fulfillment context, these
            // obligations could in principle be ignored: they have
            // already been registered when the cache entry was
            // created (and hence the new ones will quickly be
            // discarded as duplicated). But when doing trait
            // evaluation this is not the case, and dropping the trait
            // evaluations can causes ICEs (e.g., #43132).
            debug!(?ty, "found normalized ty");
            obligations.extend(ty.obligations);
            return Ok(Some(ty.value));
        }
        Err(ProjectionCacheEntry::Error) => {
            debug!("opt_normalize_projection_type: found error");
            let result = normalize_to_error(selcx, param_env, projection_term, cause, depth);
            obligations.extend(result.obligations);
            return Ok(Some(result.value));
        }
    }

    let obligation =
        Obligation::with_depth(selcx.tcx(), cause.clone(), depth, param_env, projection_term);

    match project(selcx, &obligation) {
        Ok(Projected::Progress(Progress {
            term: projected_term,
            obligations: mut projected_obligations,
        })) => {
            // if projection succeeded, then what we get out of this
            // is also non-normalized (consider: it was derived from
            // an impl, where-clause etc) and hence we must
            // re-normalize it

            let projected_term = selcx.infcx.resolve_vars_if_possible(projected_term);

            let mut result = if projected_term.has_aliases() {
                let normalized_ty = normalize_with_depth_to(
                    selcx,
                    param_env,
                    cause,
                    depth + 1,
                    projected_term,
                    &mut projected_obligations,
                );

                Normalized { value: normalized_ty, obligations: projected_obligations }
            } else {
                Normalized { value: projected_term, obligations: projected_obligations }
            };

            let mut deduped = SsoHashSet::with_capacity(result.obligations.len());
            result.obligations.retain(|obligation| deduped.insert(obligation.clone()));

            infcx.inner.borrow_mut().projection_cache().insert_term(cache_key, result.clone());
            obligations.extend(result.obligations);
            Ok(Some(result.value))
        }
        Ok(Projected::NoProgress(projected_ty)) => {
            let result =
                Normalized { value: projected_ty, obligations: PredicateObligations::new() };
            infcx.inner.borrow_mut().projection_cache().insert_term(cache_key, result.clone());
            // No need to extend `obligations`.
            Ok(Some(result.value))
        }
        Err(ProjectionError::TooManyCandidates) => {
            debug!("opt_normalize_projection_type: too many candidates");
            infcx.inner.borrow_mut().projection_cache().ambiguous(cache_key);
            Ok(None)
        }
        Err(ProjectionError::TraitSelectionError(_)) => {
            debug!("opt_normalize_projection_type: ERROR");
            // if we got an error processing the `T as Trait` part,
            // just return `ty::err` but add the obligation `T :
            // Trait`, which when processed will cause the error to be
            // reported later
            infcx.inner.borrow_mut().projection_cache().error(cache_key);
            let result = normalize_to_error(selcx, param_env, projection_term, cause, depth);
            obligations.extend(result.obligations);
            Ok(Some(result.value))
        }
    }
}

/// If we are projecting `<T as Trait>::Item`, but `T: Trait` does not
/// hold. In various error cases, we cannot generate a valid
/// normalized projection. Therefore, we create an inference variable
/// return an associated obligation that, when fulfilled, will lead to
/// an error.
///
/// Note that we used to return `Error` here, but that was quite
/// dubious -- the premise was that an error would *eventually* be
/// reported, when the obligation was processed. But in general once
/// you see an `Error` you are supposed to be able to assume that an
/// error *has been* reported, so that you can take whatever heuristic
/// paths you want to take. To make things worse, it was possible for
/// cycles to arise, where you basically had a setup like `<MyType<$0>
/// as Trait>::Foo == $0`. Here, normalizing `<MyType<$0> as
/// Trait>::Foo>` to `[type error]` would lead to an obligation of
/// `<MyType<[type error]> as Trait>::Foo`. We are supposed to report
/// an error for this obligation, but we legitimately should not,
/// because it contains `[type error]`. Yuck! (See issue #29857 for
/// one case where this arose.)
fn normalize_to_error<'a, 'tcx>(
    selcx: &SelectionContext<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_term: ty::AliasTerm<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
) -> NormalizedTerm<'tcx> {
    let trait_ref = ty::Binder::dummy(projection_term.trait_ref(selcx.tcx()));
    let new_value = match projection_term.kind(selcx.tcx()) {
        ty::AliasTermKind::ProjectionTy
        | ty::AliasTermKind::InherentTy
        | ty::AliasTermKind::OpaqueTy
        | ty::AliasTermKind::WeakTy => selcx.infcx.next_ty_var(cause.span).into(),
        ty::AliasTermKind::UnevaluatedConst | ty::AliasTermKind::ProjectionConst => {
            selcx.infcx.next_const_var(cause.span).into()
        }
    };
    let mut obligations = PredicateObligations::new();
    obligations.push(Obligation {
        cause,
        recursion_depth: depth,
        param_env,
        predicate: trait_ref.upcast(selcx.tcx()),
    });
    Normalized { value: new_value, obligations }
}

/// Confirm and normalize the given inherent projection.
#[instrument(level = "debug", skip(selcx, param_env, cause, obligations))]
pub fn normalize_inherent_projection<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    alias_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut PredicateObligations<'tcx>,
) -> Ty<'tcx> {
    let tcx = selcx.tcx();

    if !tcx.recursion_limit().value_within_limit(depth) {
        // Halt compilation because it is important that overflows never be masked.
        tcx.dcx().emit_fatal(InherentProjectionNormalizationOverflow {
            span: cause.span,
            ty: alias_ty.to_string(),
        });
    }

    let args = compute_inherent_assoc_ty_args(
        selcx,
        param_env,
        alias_ty,
        cause.clone(),
        depth,
        obligations,
    );

    // Register the obligations arising from the impl and from the associated type itself.
    let predicates = tcx.predicates_of(alias_ty.def_id).instantiate(tcx, args);
    for (predicate, span) in predicates {
        let predicate = normalize_with_depth_to(
            selcx,
            param_env,
            cause.clone(),
            depth + 1,
            predicate,
            obligations,
        );

        let nested_cause = ObligationCause::new(
            cause.span,
            cause.body_id,
            // FIXME(inherent_associated_types): Since we can't pass along the self type to the
            // cause code, inherent projections will be printed with identity instantiation in
            // diagnostics which is not ideal.
            // Consider creating separate cause codes for this specific situation.
            ObligationCauseCode::WhereClause(alias_ty.def_id, span),
        );

        obligations.push(Obligation::with_depth(
            tcx,
            nested_cause,
            depth + 1,
            param_env,
            predicate,
        ));
    }

    let ty = tcx.type_of(alias_ty.def_id).instantiate(tcx, args);

    let mut ty = selcx.infcx.resolve_vars_if_possible(ty);
    if ty.has_aliases() {
        ty = normalize_with_depth_to(selcx, param_env, cause.clone(), depth + 1, ty, obligations);
    }

    ty
}

pub fn compute_inherent_assoc_ty_args<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    alias_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut PredicateObligations<'tcx>,
) -> ty::GenericArgsRef<'tcx> {
    let tcx = selcx.tcx();

    let impl_def_id = tcx.parent(alias_ty.def_id);
    let impl_args = selcx.infcx.fresh_args_for_item(cause.span, impl_def_id);

    let mut impl_ty = tcx.type_of(impl_def_id).instantiate(tcx, impl_args);
    if !selcx.infcx.next_trait_solver() {
        impl_ty = normalize_with_depth_to(
            selcx,
            param_env,
            cause.clone(),
            depth + 1,
            impl_ty,
            obligations,
        );
    }

    // Infer the generic parameters of the impl by unifying the
    // impl type with the self type of the projection.
    let mut self_ty = alias_ty.self_ty();
    if !selcx.infcx.next_trait_solver() {
        self_ty = normalize_with_depth_to(
            selcx,
            param_env,
            cause.clone(),
            depth + 1,
            self_ty,
            obligations,
        );
    }

    match selcx.infcx.at(&cause, param_env).eq(DefineOpaqueTypes::Yes, impl_ty, self_ty) {
        Ok(mut ok) => obligations.append(&mut ok.obligations),
        Err(_) => {
            tcx.dcx().span_bug(
                cause.span,
                format!("{self_ty:?} was equal to {impl_ty:?} during selection but now it is not"),
            );
        }
    }

    alias_ty.rebase_inherent_args_onto_impl(impl_args, tcx)
}

enum Projected<'tcx> {
    Progress(Progress<'tcx>),
    NoProgress(ty::Term<'tcx>),
}

struct Progress<'tcx> {
    term: ty::Term<'tcx>,
    obligations: PredicateObligations<'tcx>,
}

impl<'tcx> Progress<'tcx> {
    fn error(tcx: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Self {
        Progress { term: Ty::new_error(tcx, guar).into(), obligations: PredicateObligations::new() }
    }

    fn with_addl_obligations(mut self, mut obligations: PredicateObligations<'tcx>) -> Self {
        self.obligations.append(&mut obligations);
        self
    }
}

/// Computes the result of a projection type (if we can).
///
/// IMPORTANT:
/// - `obligation` must be fully normalized
#[instrument(level = "info", skip(selcx))]
fn project<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
) -> Result<Projected<'tcx>, ProjectionError<'tcx>> {
    if !selcx.tcx().recursion_limit().value_within_limit(obligation.recursion_depth) {
        // This should really be an immediate error, but some existing code
        // relies on being able to recover from this.
        return Err(ProjectionError::TraitSelectionError(SelectionError::Overflow(
            OverflowError::Canonical,
        )));
    }

    if let Err(guar) = obligation.predicate.error_reported() {
        return Ok(Projected::Progress(Progress::error(selcx.tcx(), guar)));
    }

    let mut candidates = ProjectionCandidateSet::None;

    // Make sure that the following procedures are kept in order. ParamEnv
    // needs to be first because it has highest priority, and Select checks
    // the return value of push_candidate which assumes it's ran at last.
    assemble_candidates_from_param_env(selcx, obligation, &mut candidates);

    assemble_candidates_from_trait_def(selcx, obligation, &mut candidates);

    assemble_candidates_from_object_ty(selcx, obligation, &mut candidates);

    if let ProjectionCandidateSet::Single(ProjectionCandidate::Object(_)) = candidates {
        // Avoid normalization cycle from selection (see
        // `assemble_candidates_from_object_ty`).
        // FIXME(lazy_normalization): Lazy normalization should save us from
        // having to special case this.
    } else {
        assemble_candidates_from_impls(selcx, obligation, &mut candidates);
    };

    match candidates {
        ProjectionCandidateSet::Single(candidate) => {
            Ok(Projected::Progress(confirm_candidate(selcx, obligation, candidate)))
        }
        ProjectionCandidateSet::None => {
            let tcx = selcx.tcx();
            let term = match tcx.def_kind(obligation.predicate.def_id) {
                DefKind::AssocTy => Ty::new_projection_from_args(
                    tcx,
                    obligation.predicate.def_id,
                    obligation.predicate.args,
                )
                .into(),
                DefKind::AssocConst => ty::Const::new_unevaluated(
                    tcx,
                    ty::UnevaluatedConst::new(
                        obligation.predicate.def_id,
                        obligation.predicate.args,
                    ),
                )
                .into(),
                kind => {
                    bug!("unknown projection def-id: {}", kind.descr(obligation.predicate.def_id))
                }
            };

            Ok(Projected::NoProgress(term))
        }
        // Error occurred while trying to processing impls.
        ProjectionCandidateSet::Error(e) => Err(ProjectionError::TraitSelectionError(e)),
        // Inherent ambiguity that prevents us from even enumerating the
        // candidates.
        ProjectionCandidateSet::Ambiguous => Err(ProjectionError::TooManyCandidates),
    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionCandidate::ParamEnv,
        obligation.param_env.caller_bounds().iter(),
        false,
    );
}

/// In the case of a nested projection like `<<A as Foo>::FooT as Bar>::BarT`, we may find
/// that the definition of `Foo` has some clues:
///
/// ```ignore (illustrative)
/// trait Foo {
///     type FooT : Bar<BarT=i32>
/// }
/// ```
///
/// Here, for example, we could conclude that the result is `i32`.
fn assemble_candidates_from_trait_def<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_trait_def(..)");
    let mut ambiguous = false;
    selcx.for_each_item_bound(
        obligation.predicate.self_ty(),
        |selcx, clause, _| {
            let Some(clause) = clause.as_projection_clause() else {
                return ControlFlow::Continue(());
            };
            if clause.item_def_id() != obligation.predicate.def_id {
                return ControlFlow::Continue(());
            }

            let is_match =
                selcx.infcx.probe(|_| selcx.match_projection_projections(obligation, clause, true));

            match is_match {
                ProjectionMatchesProjection::Yes => {
                    candidate_set.push_candidate(ProjectionCandidate::TraitDef(clause));

                    if !obligation.predicate.has_non_region_infer() {
                        // HACK: Pick the first trait def candidate for a fully
                        // inferred predicate. This is to allow duplicates that
                        // differ only in normalization.
                        return ControlFlow::Break(());
                    }
                }
                ProjectionMatchesProjection::Ambiguous => {
                    candidate_set.mark_ambiguous();
                }
                ProjectionMatchesProjection::No => {}
            }

            ControlFlow::Continue(())
        },
        // `ProjectionCandidateSet` is borrowed in the above closure,
        // so just mark ambiguous outside of the closure.
        || ambiguous = true,
    );

    if ambiguous {
        candidate_set.mark_ambiguous();
    }
}

/// In the case of a trait object like
/// `<dyn Iterator<Item = ()> as Iterator>::Item` we can use the existential
/// predicate in the trait object.
///
/// We don't go through the select candidate for these bounds to avoid cycles:
/// In the above case, `dyn Iterator<Item = ()>: Iterator` would create a
/// nested obligation of `<dyn Iterator<Item = ()> as Iterator>::Item: Sized`,
/// this then has to be normalized without having to prove
/// `dyn Iterator<Item = ()>: Iterator` again.
fn assemble_candidates_from_object_ty<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_object_ty(..)");

    let tcx = selcx.tcx();

    if !tcx.trait_def(obligation.predicate.trait_def_id(tcx)).implement_via_object {
        return;
    }

    let self_ty = obligation.predicate.self_ty();
    let object_ty = selcx.infcx.shallow_resolve(self_ty);
    let data = match object_ty.kind() {
        ty::Dynamic(data, ..) => data,
        ty::Infer(ty::TyVar(_)) => {
            // If the self-type is an inference variable, then it MAY wind up
            // being an object type, so induce an ambiguity.
            candidate_set.mark_ambiguous();
            return;
        }
        _ => return,
    };
    let env_predicates = data
        .projection_bounds()
        .filter(|bound| bound.item_def_id() == obligation.predicate.def_id)
        .map(|p| p.with_self_ty(tcx, object_ty).upcast(tcx));

    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionCandidate::Object,
        env_predicates,
        false,
    );

    // `dyn Trait` automagically project their AFITs to `dyn* Future`.
    if tcx.is_impl_trait_in_trait(obligation.predicate.def_id)
        && let Some(out_trait_def_id) = data.principal_def_id()
        && let rpitit_trait_def_id = tcx.parent(obligation.predicate.def_id)
        && tcx
            .supertrait_def_ids(out_trait_def_id)
            .any(|trait_def_id| trait_def_id == rpitit_trait_def_id)
    {
        candidate_set.push_candidate(ProjectionCandidate::ObjectRpitit);
    }
}

#[instrument(
    level = "debug",
    skip(selcx, candidate_set, ctor, env_predicates, potentially_unnormalized_candidates)
)]
fn assemble_candidates_from_predicates<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
    ctor: fn(ty::PolyProjectionPredicate<'tcx>) -> ProjectionCandidate<'tcx>,
    env_predicates: impl Iterator<Item = ty::Clause<'tcx>>,
    potentially_unnormalized_candidates: bool,
) {
    let infcx = selcx.infcx;
    let drcx = DeepRejectCtxt::relate_rigid_rigid(selcx.tcx());
    for predicate in env_predicates {
        let bound_predicate = predicate.kind();
        if let ty::ClauseKind::Projection(data) = predicate.kind().skip_binder() {
            let data = bound_predicate.rebind(data);
            if data.item_def_id() != obligation.predicate.def_id {
                continue;
            }

            if !drcx
                .args_may_unify(obligation.predicate.args, data.skip_binder().projection_term.args)
            {
                continue;
            }

            let is_match = infcx.probe(|_| {
                selcx.match_projection_projections(
                    obligation,
                    data,
                    potentially_unnormalized_candidates,
                )
            });

            match is_match {
                ProjectionMatchesProjection::Yes => {
                    candidate_set.push_candidate(ctor(data));

                    if potentially_unnormalized_candidates
                        && !obligation.predicate.has_non_region_infer()
                    {
                        // HACK: Pick the first trait def candidate for a fully
                        // inferred predicate. This is to allow duplicates that
                        // differ only in normalization.
                        return;
                    }
                }
                ProjectionMatchesProjection::Ambiguous => {
                    candidate_set.mark_ambiguous();
                }
                ProjectionMatchesProjection::No => {}
            }
        }
    }
}

#[instrument(level = "debug", skip(selcx, obligation, candidate_set))]
fn assemble_candidates_from_impls<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let trait_ref = obligation.predicate.trait_ref(selcx.tcx());
    let trait_obligation = obligation.with(selcx.tcx(), trait_ref);
    let _ = selcx.infcx.commit_if_ok(|_| {
        let impl_source = match selcx.select(&trait_obligation) {
            Ok(Some(impl_source)) => impl_source,
            Ok(None) => {
                candidate_set.mark_ambiguous();
                return Err(());
            }
            Err(e) => {
                debug!(error = ?e, "selection error");
                candidate_set.mark_error(e);
                return Err(());
            }
        };

        let eligible = match &impl_source {
            ImplSource::UserDefined(impl_data) => {
                // We have to be careful when projecting out of an
                // impl because of specialization. If we are not in
                // codegen (i.e., projection mode is not "any"), and the
                // impl's type is declared as default, then we disable
                // projection (even if the trait ref is fully
                // monomorphic). In the case where trait ref is not
                // fully monomorphic (i.e., includes type parameters),
                // this is because those type parameters may
                // ultimately be bound to types from other crates that
                // may have specialized impls we can't see. In the
                // case where the trait ref IS fully monomorphic, this
                // is a policy decision that we made in the RFC in
                // order to preserve flexibility for the crate that
                // defined the specializable impl to specialize later
                // for existing types.
                //
                // In either case, we handle this by not adding a
                // candidate for an impl if it contains a `default`
                // type.
                //
                // NOTE: This should be kept in sync with the similar code in
                // `rustc_ty_utils::instance::resolve_associated_item()`.
                let node_item = specialization_graph::assoc_def(
                    selcx.tcx(),
                    impl_data.impl_def_id,
                    obligation.predicate.def_id,
                )
                .map_err(|ErrorGuaranteed { .. }| ())?;

                if node_item.is_final() {
                    // Non-specializable items are always projectable.
                    true
                } else {
                    // Only reveal a specializable default if we're past type-checking
                    // and the obligation is monomorphic, otherwise passes such as
                    // transmute checking and polymorphic MIR optimizations could
                    // get a result which isn't correct for all monomorphizations.
                    match selcx.infcx.typing_mode() {
                        TypingMode::Coherence
                        | TypingMode::Analysis { .. }
                        | TypingMode::PostBorrowckAnalysis { .. } => {
                            debug!(
                                assoc_ty = ?selcx.tcx().def_path_str(node_item.item.def_id),
                                ?obligation.predicate,
                                "assemble_candidates_from_impls: not eligible due to default",
                            );
                            false
                        }
                        TypingMode::PostAnalysis => {
                            // NOTE(eddyb) inference variables can resolve to parameters, so
                            // assume `poly_trait_ref` isn't monomorphic, if it contains any.
                            let poly_trait_ref = selcx.infcx.resolve_vars_if_possible(trait_ref);
                            !poly_trait_ref.still_further_specializable()
                        }
                    }
                }
            }
            ImplSource::Builtin(BuiltinImplSource::Misc, _) => {
                // While a builtin impl may be known to exist, the associated type may not yet
                // be known. Any type with multiple potential associated types is therefore
                // not eligible.
                let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());

                let tcx = selcx.tcx();
                let lang_items = selcx.tcx().lang_items();
                if [
                    lang_items.coroutine_trait(),
                    lang_items.future_trait(),
                    lang_items.iterator_trait(),
                    lang_items.async_iterator_trait(),
                    lang_items.fn_trait(),
                    lang_items.fn_mut_trait(),
                    lang_items.fn_once_trait(),
                    lang_items.async_fn_trait(),
                    lang_items.async_fn_mut_trait(),
                    lang_items.async_fn_once_trait(),
                ]
                .contains(&Some(trait_ref.def_id))
                {
                    true
                } else if tcx.is_lang_item(trait_ref.def_id, LangItem::AsyncFnKindHelper) {
                    // FIXME(async_closures): Validity constraints here could be cleaned up.
                    if obligation.predicate.args.type_at(0).is_ty_var()
                        || obligation.predicate.args.type_at(4).is_ty_var()
                        || obligation.predicate.args.type_at(5).is_ty_var()
                    {
                        candidate_set.mark_ambiguous();
                        true
                    } else {
                        obligation.predicate.args.type_at(0).to_opt_closure_kind().is_some()
                            && obligation.predicate.args.type_at(1).to_opt_closure_kind().is_some()
                    }
                } else if tcx.is_lang_item(trait_ref.def_id, LangItem::DiscriminantKind) {
                    match self_ty.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Adt(..)
                        | ty::Foreign(_)
                        | ty::Str
                        | ty::Array(..)
                        | ty::Pat(..)
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::CoroutineClosure(..)
                        | ty::Coroutine(..)
                        | ty::CoroutineWitness(..)
                        | ty::Never
                        | ty::Tuple(..)
                        // Integers and floats always have `u8` as their discriminant.
                        | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                        // type parameters, opaques, and unnormalized projections don't have
                        // a known discriminant and may need to be normalized further or rely
                        // on param env for discriminant projections
                        ty::Param(_)
                        | ty::Alias(..)
                        | ty::Bound(..)
                        | ty::Placeholder(..)
                        | ty::Infer(..)
                        | ty::Error(_) => false,
                    }
                } else if tcx.is_lang_item(trait_ref.def_id, LangItem::AsyncDestruct) {
                    match self_ty.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Adt(..)
                        | ty::Str
                        | ty::Array(..)
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::CoroutineClosure(..)
                        | ty::Coroutine(..)
                        | ty::CoroutineWitness(..)
                        | ty::Pat(..)
                        | ty::Never
                        | ty::Tuple(..)
                        | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                        // type parameters, opaques, and unnormalized projections don't have
                        // a known async destructor and may need to be normalized further or rely
                        // on param env for async destructor projections
                        ty::Param(_)
                        | ty::Foreign(_)
                        | ty::Alias(..)
                        | ty::Bound(..)
                        | ty::Placeholder(..)
                        | ty::Infer(_)
                        | ty::Error(_) => false,
                    }
                } else if tcx.is_lang_item(trait_ref.def_id, LangItem::PointeeTrait) {
                    let tail = selcx.tcx().struct_tail_raw(
                        self_ty,
                        |ty| {
                            // We throw away any obligations we get from this, since we normalize
                            // and confirm these obligations once again during confirmation
                            normalize_with_depth(
                                selcx,
                                obligation.param_env,
                                obligation.cause.clone(),
                                obligation.recursion_depth + 1,
                                ty,
                            )
                            .value
                        },
                        || {},
                    );

                    match tail.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Str
                        | ty::Array(..)
                        | ty::Pat(..)
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::CoroutineClosure(..)
                        | ty::Coroutine(..)
                        | ty::CoroutineWitness(..)
                        | ty::Never
                        // Extern types have unit metadata, according to RFC 2850
                        | ty::Foreign(_)
                        // If returned by `struct_tail` this is a unit struct
                        // without any fields, or not a struct, and therefore is Sized.
                        | ty::Adt(..)
                        // If returned by `struct_tail` this is the empty tuple.
                        | ty::Tuple(..)
                        // Integers and floats are always Sized, and so have unit type metadata.
                        | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                        // We normalize from `Wrapper<Tail>::Metadata` to `Tail::Metadata` if able.
                        // Otherwise, type parameters, opaques, and unnormalized projections have
                        // unit metadata if they're known (e.g. by the param_env) to be sized.
                        ty::Param(_) | ty::Alias(..)
                            if self_ty != tail
                                || selcx.infcx.predicate_must_hold_modulo_regions(
                                    &obligation.with(
                                        selcx.tcx(),
                                        ty::TraitRef::new(
                                            selcx.tcx(),
                                            selcx.tcx().require_lang_item(
                                                LangItem::Sized,
                                                Some(obligation.cause.span),
                                            ),
                                            [self_ty],
                                        ),
                                    ),
                                ) =>
                        {
                            true
                        }

                        // FIXME(compiler-errors): are Bound and Placeholder types ever known sized?
                        ty::Param(_)
                        | ty::Alias(..)
                        | ty::Bound(..)
                        | ty::Placeholder(..)
                        | ty::Infer(..)
                        | ty::Error(_) => {
                            if tail.has_infer_types() {
                                candidate_set.mark_ambiguous();
                            }
                            false
                        }
                    }
                } else if tcx.trait_is_auto(trait_ref.def_id) {
                    tcx.dcx().span_delayed_bug(
                        tcx.def_span(obligation.predicate.def_id),
                        "associated types not allowed on auto traits",
                    );
                    false
                } else {
                    bug!("unexpected builtin trait with associated type: {trait_ref:?}")
                }
            }
            ImplSource::Param(..) => {
                // This case tell us nothing about the value of an
                // associated type. Consider:
                //
                // ```
                // trait SomeTrait { type Foo; }
                // fn foo<T:SomeTrait>(...) { }
                // ```
                //
                // If the user writes `<T as SomeTrait>::Foo`, then the `T
                // : SomeTrait` binding does not help us decide what the
                // type `Foo` is (at least, not more specifically than
                // what we already knew).
                //
                // But wait, you say! What about an example like this:
                //
                // ```
                // fn bar<T:SomeTrait<Foo=usize>>(...) { ... }
                // ```
                //
                // Doesn't the `T : SomeTrait<Foo=usize>` predicate help
                // resolve `T::Foo`? And of course it does, but in fact
                // that single predicate is desugared into two predicates
                // in the compiler: a trait predicate (`T : SomeTrait`) and a
                // projection. And the projection where clause is handled
                // in `assemble_candidates_from_param_env`.
                false
            }
            ImplSource::Builtin(BuiltinImplSource::Object { .. }, _) => {
                // Handled by the `Object` projection candidate. See
                // `assemble_candidates_from_object_ty` for an explanation of
                // why we special case object types.
                false
            }
            ImplSource::Builtin(BuiltinImplSource::TraitUpcasting { .. }, _)
            | ImplSource::Builtin(BuiltinImplSource::TupleUnsizing, _) => {
                // These traits have no associated types.
                selcx.tcx().dcx().span_delayed_bug(
                    obligation.cause.span,
                    format!("Cannot project an associated type from `{impl_source:?}`"),
                );
                return Err(());
            }
        };

        if eligible {
            if candidate_set.push_candidate(ProjectionCandidate::Select(impl_source)) {
                Ok(())
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    });
}

fn confirm_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    candidate: ProjectionCandidate<'tcx>,
) -> Progress<'tcx> {
    debug!(?obligation, ?candidate, "confirm_candidate");
    let mut progress = match candidate {
        ProjectionCandidate::ParamEnv(poly_projection)
        | ProjectionCandidate::Object(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection, false)
        }

        ProjectionCandidate::TraitDef(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection, true)
        }

        ProjectionCandidate::Select(impl_source) => {
            confirm_select_candidate(selcx, obligation, impl_source)
        }

        ProjectionCandidate::ObjectRpitit => confirm_object_rpitit_candidate(selcx, obligation),
    };

    // When checking for cycle during evaluation, we compare predicates with
    // "syntactic" equality. Since normalization generally introduces a type
    // with new region variables, we need to resolve them to existing variables
    // when possible for this to work. See `auto-trait-projection-recursion.rs`
    // for a case where this matters.
    if progress.term.has_infer_regions() {
        progress.term = progress.term.fold_with(&mut OpportunisticRegionResolver::new(selcx.infcx));
    }
    progress
}

fn confirm_select_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    impl_source: Selection<'tcx>,
) -> Progress<'tcx> {
    match impl_source {
        ImplSource::UserDefined(data) => confirm_impl_candidate(selcx, obligation, data),
        ImplSource::Builtin(BuiltinImplSource::Misc, data) => {
            let tcx = selcx.tcx();
            let trait_def_id = obligation.predicate.trait_def_id(tcx);
            if tcx.is_lang_item(trait_def_id, LangItem::Coroutine) {
                confirm_coroutine_candidate(selcx, obligation, data)
            } else if tcx.is_lang_item(trait_def_id, LangItem::Future) {
                confirm_future_candidate(selcx, obligation, data)
            } else if tcx.is_lang_item(trait_def_id, LangItem::Iterator) {
                confirm_iterator_candidate(selcx, obligation, data)
            } else if tcx.is_lang_item(trait_def_id, LangItem::AsyncIterator) {
                confirm_async_iterator_candidate(selcx, obligation, data)
            } else if selcx.tcx().fn_trait_kind_from_def_id(trait_def_id).is_some() {
                if obligation.predicate.self_ty().is_closure()
                    || obligation.predicate.self_ty().is_coroutine_closure()
                {
                    confirm_closure_candidate(selcx, obligation, data)
                } else {
                    confirm_fn_pointer_candidate(selcx, obligation, data)
                }
            } else if selcx.tcx().async_fn_trait_kind_from_def_id(trait_def_id).is_some() {
                confirm_async_closure_candidate(selcx, obligation, data)
            } else if tcx.is_lang_item(trait_def_id, LangItem::AsyncFnKindHelper) {
                confirm_async_fn_kind_helper_candidate(selcx, obligation, data)
            } else {
                confirm_builtin_candidate(selcx, obligation, data)
            }
        }
        ImplSource::Builtin(BuiltinImplSource::Object { .. }, _)
        | ImplSource::Param(..)
        | ImplSource::Builtin(BuiltinImplSource::TraitUpcasting { .. }, _)
        | ImplSource::Builtin(BuiltinImplSource::TupleUnsizing, _) => {
            // we don't create Select candidates with this kind of resolution
            span_bug!(
                obligation.cause.span,
                "Cannot project an associated type from `{:?}`",
                impl_source
            )
        }
    }
}

fn confirm_coroutine_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());
    let ty::Coroutine(_, args) = self_ty.kind() else {
        unreachable!(
            "expected coroutine self type for built-in coroutine candidate, found {self_ty}"
        )
    };
    let coroutine_sig = args.as_coroutine().sig();
    let Normalized { value: coroutine_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        coroutine_sig,
    );

    debug!(?obligation, ?coroutine_sig, ?obligations, "confirm_coroutine_candidate");

    let tcx = selcx.tcx();

    let coroutine_def_id = tcx.require_lang_item(LangItem::Coroutine, None);

    let (trait_ref, yield_ty, return_ty) = super::util::coroutine_trait_ref_and_outputs(
        tcx,
        coroutine_def_id,
        obligation.predicate.self_ty(),
        coroutine_sig,
    );

    let ty = if tcx.is_lang_item(obligation.predicate.def_id, LangItem::CoroutineReturn) {
        return_ty
    } else if tcx.is_lang_item(obligation.predicate.def_id, LangItem::CoroutineYield) {
        yield_ty
    } else {
        span_bug!(
            tcx.def_span(obligation.predicate.def_id),
            "unexpected associated type: `Coroutine::{}`",
            tcx.item_name(obligation.predicate.def_id),
        );
    };

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(
            tcx,
            obligation.predicate.def_id,
            trait_ref.args,
        ),
        term: ty.into(),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_future_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());
    let ty::Coroutine(_, args) = self_ty.kind() else {
        unreachable!(
            "expected coroutine self type for built-in async future candidate, found {self_ty}"
        )
    };
    let coroutine_sig = args.as_coroutine().sig();
    let Normalized { value: coroutine_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        coroutine_sig,
    );

    debug!(?obligation, ?coroutine_sig, ?obligations, "confirm_future_candidate");

    let tcx = selcx.tcx();
    let fut_def_id = tcx.require_lang_item(LangItem::Future, None);

    let (trait_ref, return_ty) = super::util::future_trait_ref_and_outputs(
        tcx,
        fut_def_id,
        obligation.predicate.self_ty(),
        coroutine_sig,
    );

    debug_assert_eq!(tcx.associated_item(obligation.predicate.def_id).name, sym::Output);

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(
            tcx,
            obligation.predicate.def_id,
            trait_ref.args,
        ),
        term: return_ty.into(),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_iterator_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());
    let ty::Coroutine(_, args) = self_ty.kind() else {
        unreachable!("expected coroutine self type for built-in gen candidate, found {self_ty}")
    };
    let gen_sig = args.as_coroutine().sig();
    let Normalized { value: gen_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        gen_sig,
    );

    debug!(?obligation, ?gen_sig, ?obligations, "confirm_iterator_candidate");

    let tcx = selcx.tcx();
    let iter_def_id = tcx.require_lang_item(LangItem::Iterator, None);

    let (trait_ref, yield_ty) = super::util::iterator_trait_ref_and_outputs(
        tcx,
        iter_def_id,
        obligation.predicate.self_ty(),
        gen_sig,
    );

    debug_assert_eq!(tcx.associated_item(obligation.predicate.def_id).name, sym::Item);

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(
            tcx,
            obligation.predicate.def_id,
            trait_ref.args,
        ),
        term: yield_ty.into(),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_async_iterator_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let ty::Coroutine(_, args) = selcx.infcx.shallow_resolve(obligation.predicate.self_ty()).kind()
    else {
        unreachable!()
    };
    let gen_sig = args.as_coroutine().sig();
    let Normalized { value: gen_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        gen_sig,
    );

    debug!(?obligation, ?gen_sig, ?obligations, "confirm_async_iterator_candidate");

    let tcx = selcx.tcx();
    let iter_def_id = tcx.require_lang_item(LangItem::AsyncIterator, None);

    let (trait_ref, yield_ty) = super::util::async_iterator_trait_ref_and_outputs(
        tcx,
        iter_def_id,
        obligation.predicate.self_ty(),
        gen_sig,
    );

    debug_assert_eq!(tcx.associated_item(obligation.predicate.def_id).name, sym::Item);

    let ty::Adt(_poll_adt, args) = *yield_ty.kind() else {
        bug!();
    };
    let ty::Adt(_option_adt, args) = *args.type_at(0).kind() else {
        bug!();
    };
    let item_ty = args.type_at(0);

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(
            tcx,
            obligation.predicate.def_id,
            trait_ref.args,
        ),
        term: item_ty.into(),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_builtin_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    data: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let self_ty = obligation.predicate.self_ty();
    let item_def_id = obligation.predicate.def_id;
    let trait_def_id = tcx.trait_of_item(item_def_id).unwrap();
    let args = tcx.mk_args(&[self_ty.into()]);
    let (term, obligations) = if tcx.is_lang_item(trait_def_id, LangItem::DiscriminantKind) {
        let discriminant_def_id = tcx.require_lang_item(LangItem::Discriminant, None);
        assert_eq!(discriminant_def_id, item_def_id);

        (self_ty.discriminant_ty(tcx).into(), PredicateObligations::new())
    } else if tcx.is_lang_item(trait_def_id, LangItem::AsyncDestruct) {
        let destructor_def_id = tcx.associated_item_def_ids(trait_def_id)[0];
        assert_eq!(destructor_def_id, item_def_id);

        (self_ty.async_destructor_ty(tcx).into(), PredicateObligations::new())
    } else if tcx.is_lang_item(trait_def_id, LangItem::PointeeTrait) {
        let metadata_def_id = tcx.require_lang_item(LangItem::Metadata, None);
        assert_eq!(metadata_def_id, item_def_id);

        let mut obligations = PredicateObligations::new();
        let normalize = |ty| {
            normalize_with_depth_to(
                selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                ty,
                &mut obligations,
            )
        };
        let metadata_ty = self_ty.ptr_metadata_ty_or_tail(tcx, normalize).unwrap_or_else(|tail| {
            if tail == self_ty {
                // This is the "fallback impl" for type parameters, unnormalizable projections
                // and opaque types: If the `self_ty` is `Sized`, then the metadata is `()`.
                // FIXME(ptr_metadata): This impl overlaps with the other impls and shouldn't
                // exist. Instead, `Pointee<Metadata = ()>` should be a supertrait of `Sized`.
                let sized_predicate = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::Sized, Some(obligation.cause.span)),
                    [self_ty],
                );
                obligations.push(obligation.with(tcx, sized_predicate));
                tcx.types.unit
            } else {
                // We know that `self_ty` has the same metadata as `tail`. This allows us
                // to prove predicates like `Wrapper<Tail>::Metadata == Tail::Metadata`.
                Ty::new_projection(tcx, metadata_def_id, [tail])
            }
        });
        (metadata_ty.into(), obligations)
    } else {
        bug!("unexpected builtin trait with associated type: {:?}", obligation.predicate);
    };

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(tcx, item_def_id, args),
        term,
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(obligations)
        .with_addl_obligations(data)
}

fn confirm_fn_pointer_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let fn_type = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());
    let sig = fn_type.fn_sig(tcx);
    let Normalized { value: sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        sig,
    );

    confirm_callable_candidate(selcx, obligation, sig, util::TupleArgumentsFlag::Yes)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_closure_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());
    let closure_sig = match *self_ty.kind() {
        ty::Closure(_, args) => args.as_closure().sig(),

        // Construct a "normal" `FnOnce` signature for coroutine-closure. This is
        // basically duplicated with the `AsyncFnOnce::CallOnce` confirmation, but
        // I didn't see a good way to unify those.
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            args.coroutine_closure_sig().map_bound(|sig| {
                // If we know the kind and upvars, use that directly.
                // Otherwise, defer to `AsyncFnKindHelper::Upvars` to delay
                // the projection, like the `AsyncFn*` traits do.
                let output_ty = if let Some(_) = kind_ty.to_opt_closure_kind()
                    // Fall back to projection if upvars aren't constrained
                    && !args.tupled_upvars_ty().is_ty_var()
                {
                    sig.to_coroutine_given_kind_and_upvars(
                        tcx,
                        args.parent_args(),
                        tcx.coroutine_for_closure(def_id),
                        ty::ClosureKind::FnOnce,
                        tcx.lifetimes.re_static,
                        args.tupled_upvars_ty(),
                        args.coroutine_captures_by_ref_ty(),
                    )
                } else {
                    let upvars_projection_def_id =
                        tcx.require_lang_item(LangItem::AsyncFnKindUpvars, None);
                    let tupled_upvars_ty = Ty::new_projection(tcx, upvars_projection_def_id, [
                        ty::GenericArg::from(kind_ty),
                        Ty::from_closure_kind(tcx, ty::ClosureKind::FnOnce).into(),
                        tcx.lifetimes.re_static.into(),
                        sig.tupled_inputs_ty.into(),
                        args.tupled_upvars_ty().into(),
                        args.coroutine_captures_by_ref_ty().into(),
                    ]);
                    sig.to_coroutine(
                        tcx,
                        args.parent_args(),
                        Ty::from_closure_kind(tcx, ty::ClosureKind::FnOnce),
                        tcx.coroutine_for_closure(def_id),
                        tupled_upvars_ty,
                    )
                };
                tcx.mk_fn_sig(
                    [sig.tupled_inputs_ty],
                    output_ty,
                    sig.c_variadic,
                    sig.safety,
                    sig.abi,
                )
            })
        }

        _ => {
            unreachable!("expected closure self type for closure candidate, found {self_ty}");
        }
    };

    let Normalized { value: closure_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        closure_sig,
    );

    debug!(?obligation, ?closure_sig, ?obligations, "confirm_closure_candidate");

    confirm_callable_candidate(selcx, obligation, closure_sig, util::TupleArgumentsFlag::No)
        .with_addl_obligations(nested)
        .with_addl_obligations(obligations)
}

fn confirm_callable_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
    flag: util::TupleArgumentsFlag,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();

    debug!(?obligation, ?fn_sig, "confirm_callable_candidate");

    let fn_once_def_id = tcx.require_lang_item(LangItem::FnOnce, None);
    let fn_once_output_def_id = tcx.require_lang_item(LangItem::FnOnceOutput, None);

    let predicate = super::util::closure_trait_ref_and_return_type(
        tcx,
        fn_once_def_id,
        obligation.predicate.self_ty(),
        fn_sig,
        flag,
    )
    .map_bound(|(trait_ref, ret_type)| ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(tcx, fn_once_output_def_id, trait_ref.args),
        term: ret_type.into(),
    });

    confirm_param_env_candidate(selcx, obligation, predicate, true)
}

fn confirm_async_closure_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());

    let goal_kind =
        tcx.async_fn_trait_kind_from_def_id(obligation.predicate.trait_def_id(tcx)).unwrap();
    let env_region = match goal_kind {
        ty::ClosureKind::Fn | ty::ClosureKind::FnMut => obligation.predicate.args.region_at(2),
        ty::ClosureKind::FnOnce => tcx.lifetimes.re_static,
    };
    let item_name = tcx.item_name(obligation.predicate.def_id);

    let poly_cache_entry = match *self_ty.kind() {
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            let sig = args.coroutine_closure_sig().skip_binder();

            let term = match item_name {
                sym::CallOnceFuture | sym::CallRefFuture => {
                    if let Some(closure_kind) = kind_ty.to_opt_closure_kind()
                        // Fall back to projection if upvars aren't constrained
                        && !args.tupled_upvars_ty().is_ty_var()
                    {
                        if !closure_kind.extends(goal_kind) {
                            bug!("we should not be confirming if the closure kind is not met");
                        }
                        sig.to_coroutine_given_kind_and_upvars(
                            tcx,
                            args.parent_args(),
                            tcx.coroutine_for_closure(def_id),
                            goal_kind,
                            env_region,
                            args.tupled_upvars_ty(),
                            args.coroutine_captures_by_ref_ty(),
                        )
                    } else {
                        let upvars_projection_def_id =
                            tcx.require_lang_item(LangItem::AsyncFnKindUpvars, None);
                        // When we don't know the closure kind (and therefore also the closure's upvars,
                        // which are computed at the same time), we must delay the computation of the
                        // generator's upvars. We do this using the `AsyncFnKindHelper`, which as a trait
                        // goal functions similarly to the old `ClosureKind` predicate, and ensures that
                        // the goal kind <= the closure kind. As a projection `AsyncFnKindHelper::Upvars`
                        // will project to the right upvars for the generator, appending the inputs and
                        // coroutine upvars respecting the closure kind.
                        // N.B. No need to register a `AsyncFnKindHelper` goal here, it's already in `nested`.
                        let tupled_upvars_ty = Ty::new_projection(tcx, upvars_projection_def_id, [
                            ty::GenericArg::from(kind_ty),
                            Ty::from_closure_kind(tcx, goal_kind).into(),
                            env_region.into(),
                            sig.tupled_inputs_ty.into(),
                            args.tupled_upvars_ty().into(),
                            args.coroutine_captures_by_ref_ty().into(),
                        ]);
                        sig.to_coroutine(
                            tcx,
                            args.parent_args(),
                            Ty::from_closure_kind(tcx, goal_kind),
                            tcx.coroutine_for_closure(def_id),
                            tupled_upvars_ty,
                        )
                    }
                }
                sym::Output => sig.return_ty,
                name => bug!("no such associated type: {name}"),
            };
            let projection_term = match item_name {
                sym::CallOnceFuture | sym::Output => {
                    ty::AliasTerm::new(tcx, obligation.predicate.def_id, [
                        self_ty,
                        sig.tupled_inputs_ty,
                    ])
                }
                sym::CallRefFuture => ty::AliasTerm::new(tcx, obligation.predicate.def_id, [
                    ty::GenericArg::from(self_ty),
                    sig.tupled_inputs_ty.into(),
                    env_region.into(),
                ]),
                name => bug!("no such associated type: {name}"),
            };

            args.coroutine_closure_sig()
                .rebind(ty::ProjectionPredicate { projection_term, term: term.into() })
        }
        ty::FnDef(..) | ty::FnPtr(..) => {
            let bound_sig = self_ty.fn_sig(tcx);
            let sig = bound_sig.skip_binder();

            let term = match item_name {
                sym::CallOnceFuture | sym::CallRefFuture => sig.output(),
                sym::Output => {
                    let future_output_def_id = tcx.require_lang_item(LangItem::FutureOutput, None);
                    Ty::new_projection(tcx, future_output_def_id, [sig.output()])
                }
                name => bug!("no such associated type: {name}"),
            };
            let projection_term = match item_name {
                sym::CallOnceFuture | sym::Output => {
                    ty::AliasTerm::new(tcx, obligation.predicate.def_id, [
                        self_ty,
                        Ty::new_tup(tcx, sig.inputs()),
                    ])
                }
                sym::CallRefFuture => ty::AliasTerm::new(tcx, obligation.predicate.def_id, [
                    ty::GenericArg::from(self_ty),
                    Ty::new_tup(tcx, sig.inputs()).into(),
                    env_region.into(),
                ]),
                name => bug!("no such associated type: {name}"),
            };

            bound_sig.rebind(ty::ProjectionPredicate { projection_term, term: term.into() })
        }
        ty::Closure(_, args) => {
            let args = args.as_closure();
            let bound_sig = args.sig();
            let sig = bound_sig.skip_binder();

            let term = match item_name {
                sym::CallOnceFuture | sym::CallRefFuture => sig.output(),
                sym::Output => {
                    let future_output_def_id = tcx.require_lang_item(LangItem::FutureOutput, None);
                    Ty::new_projection(tcx, future_output_def_id, [sig.output()])
                }
                name => bug!("no such associated type: {name}"),
            };
            let projection_term = match item_name {
                sym::CallOnceFuture | sym::Output => {
                    ty::AliasTerm::new(tcx, obligation.predicate.def_id, [self_ty, sig.inputs()[0]])
                }
                sym::CallRefFuture => ty::AliasTerm::new(tcx, obligation.predicate.def_id, [
                    ty::GenericArg::from(self_ty),
                    sig.inputs()[0].into(),
                    env_region.into(),
                ]),
                name => bug!("no such associated type: {name}"),
            };

            bound_sig.rebind(ty::ProjectionPredicate { projection_term, term: term.into() })
        }
        _ => bug!("expected callable type for AsyncFn candidate"),
    };

    confirm_param_env_candidate(selcx, obligation, poly_cache_entry, true)
        .with_addl_obligations(nested)
}

fn confirm_async_fn_kind_helper_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: PredicateObligations<'tcx>,
) -> Progress<'tcx> {
    let [
        // We already checked that the goal_kind >= closure_kind
        _closure_kind_ty,
        goal_kind_ty,
        borrow_region,
        tupled_inputs_ty,
        tupled_upvars_ty,
        coroutine_captures_by_ref_ty,
    ] = **obligation.predicate.args
    else {
        bug!();
    };

    let predicate = ty::ProjectionPredicate {
        projection_term: ty::AliasTerm::new_from_args(
            selcx.tcx(),
            obligation.predicate.def_id,
            obligation.predicate.args,
        ),
        term: ty::CoroutineClosureSignature::tupled_upvars_by_closure_kind(
            selcx.tcx(),
            goal_kind_ty.expect_ty().to_opt_closure_kind().unwrap(),
            tupled_inputs_ty.expect_ty(),
            tupled_upvars_ty.expect_ty(),
            coroutine_captures_by_ref_ty.expect_ty(),
            borrow_region.expect_region(),
        )
        .into(),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(nested)
}

fn confirm_param_env_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    poly_cache_entry: ty::PolyProjectionPredicate<'tcx>,
    potentially_unnormalized_candidate: bool,
) -> Progress<'tcx> {
    let infcx = selcx.infcx;
    let cause = &obligation.cause;
    let param_env = obligation.param_env;

    let cache_entry = infcx.instantiate_binder_with_fresh_vars(
        cause.span,
        BoundRegionConversionTime::HigherRankedType,
        poly_cache_entry,
    );

    let cache_projection = cache_entry.projection_term;
    let mut nested_obligations = PredicateObligations::new();
    let obligation_projection = obligation.predicate;
    let obligation_projection = ensure_sufficient_stack(|| {
        normalize_with_depth_to(
            selcx,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            obligation_projection,
            &mut nested_obligations,
        )
    });
    let cache_projection = if potentially_unnormalized_candidate {
        ensure_sufficient_stack(|| {
            normalize_with_depth_to(
                selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                cache_projection,
                &mut nested_obligations,
            )
        })
    } else {
        cache_projection
    };

    debug!(?cache_projection, ?obligation_projection);

    match infcx.at(cause, param_env).eq(
        DefineOpaqueTypes::Yes,
        cache_projection,
        obligation_projection,
    ) {
        Ok(InferOk { value: _, obligations }) => {
            nested_obligations.extend(obligations);
            assoc_ty_own_obligations(selcx, obligation, &mut nested_obligations);
            // FIXME(associated_const_equality): Handle consts here as well? Maybe this progress type should just take
            // a term instead.
            Progress { term: cache_entry.term, obligations: nested_obligations }
        }
        Err(e) => {
            let msg = format!(
                "Failed to unify obligation `{obligation:?}` with poly_projection `{poly_cache_entry:?}`: {e:?}",
            );
            debug!("confirm_param_env_candidate: {}", msg);
            let err = Ty::new_error_with_message(infcx.tcx, obligation.cause.span, msg);
            Progress { term: err.into(), obligations: PredicateObligations::new() }
        }
    }
}

fn confirm_impl_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    impl_impl_source: ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();

    let ImplSourceUserDefinedData { impl_def_id, args, mut nested } = impl_impl_source;

    let assoc_item_id = obligation.predicate.def_id;
    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();

    let param_env = obligation.param_env;
    let assoc_ty = match specialization_graph::assoc_def(tcx, impl_def_id, assoc_item_id) {
        Ok(assoc_ty) => assoc_ty,
        Err(guar) => return Progress::error(tcx, guar),
    };

    if !assoc_ty.item.defaultness(tcx).has_value() {
        // This means that the impl is missing a definition for the
        // associated type. This error will be reported by the type
        // checker method `check_impl_items_against_trait`, so here we
        // just return Error.
        debug!(
            "confirm_impl_candidate: no associated type {:?} for {:?}",
            assoc_ty.item.name, obligation.predicate
        );
        return Progress { term: Ty::new_misc_error(tcx).into(), obligations: nested };
    }
    // If we're trying to normalize `<Vec<u32> as X>::A<S>` using
    //`impl<T> X for Vec<T> { type A<Y> = Box<Y>; }`, then:
    //
    // * `obligation.predicate.args` is `[Vec<u32>, S]`
    // * `args` is `[u32]`
    // * `args` ends up as `[u32, S]`
    let args = obligation.predicate.args.rebase_onto(tcx, trait_def_id, args);
    let args = translate_args(selcx.infcx, param_env, impl_def_id, args, assoc_ty.defining_node);
    let is_const = matches!(tcx.def_kind(assoc_ty.item.def_id), DefKind::AssocConst);
    let term: ty::EarlyBinder<'tcx, ty::Term<'tcx>> = if is_const {
        let did = assoc_ty.item.def_id;
        let identity_args = crate::traits::GenericArgs::identity_for_item(tcx, did);
        let uv = ty::UnevaluatedConst::new(did, identity_args);
        ty::EarlyBinder::bind(ty::Const::new_unevaluated(tcx, uv).into())
    } else {
        tcx.type_of(assoc_ty.item.def_id).map_bound(|ty| ty.into())
    };
    if !tcx.check_args_compatible(assoc_ty.item.def_id, args) {
        let err = Ty::new_error_with_message(
            tcx,
            obligation.cause.span,
            "impl item and trait item have different parameters",
        );
        Progress { term: err.into(), obligations: nested }
    } else {
        assoc_ty_own_obligations(selcx, obligation, &mut nested);
        Progress { term: term.instantiate(tcx, args), obligations: nested }
    }
}

fn confirm_object_rpitit_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let mut obligations = thin_vec![];

    // Compute an intersection lifetime for all the input components of this GAT.
    let intersection =
        selcx.infcx.next_region_var(RegionVariableOrigin::MiscVariable(obligation.cause.span));
    for component in obligation.predicate.args {
        match component.unpack() {
            ty::GenericArgKind::Lifetime(lt) => {
                obligations.push(obligation.with(tcx, ty::OutlivesPredicate(lt, intersection)));
            }
            ty::GenericArgKind::Type(ty) => {
                obligations.push(obligation.with(tcx, ty::OutlivesPredicate(ty, intersection)));
            }
            ty::GenericArgKind::Const(_ct) => {
                // Consts have no outlives...
            }
        }
    }

    Progress {
        term: Ty::new_dynamic(
            tcx,
            tcx.item_bounds_to_existential_predicates(
                obligation.predicate.def_id,
                obligation.predicate.args,
            ),
            intersection,
            ty::DynStar,
        )
        .into(),
        obligations,
    }
}

// Get obligations corresponding to the predicates from the where-clause of the
// associated type itself.
fn assoc_ty_own_obligations<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTermObligation<'tcx>,
    nested: &mut PredicateObligations<'tcx>,
) {
    let tcx = selcx.tcx();
    let predicates = tcx
        .predicates_of(obligation.predicate.def_id)
        .instantiate_own(tcx, obligation.predicate.args);
    for (predicate, span) in predicates {
        let normalized = normalize_with_depth_to(
            selcx,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            predicate,
            nested,
        );

        let nested_cause = if matches!(
            obligation.cause.code(),
            ObligationCauseCode::CompareImplItem { .. }
                | ObligationCauseCode::CheckAssociatedTypeBounds { .. }
                | ObligationCauseCode::AscribeUserTypeProvePredicate(..)
        ) {
            obligation.cause.clone()
        } else {
            ObligationCause::new(
                obligation.cause.span,
                obligation.cause.body_id,
                ObligationCauseCode::WhereClause(obligation.predicate.def_id, span),
            )
        };
        nested.push(Obligation::with_depth(
            tcx,
            nested_cause,
            obligation.recursion_depth + 1,
            obligation.param_env,
            normalized,
        ));
    }
}

pub(crate) trait ProjectionCacheKeyExt<'cx, 'tcx>: Sized {
    fn from_poly_projection_obligation(
        selcx: &mut SelectionContext<'cx, 'tcx>,
        obligation: &PolyProjectionObligation<'tcx>,
    ) -> Option<Self>;
}

impl<'cx, 'tcx> ProjectionCacheKeyExt<'cx, 'tcx> for ProjectionCacheKey<'tcx> {
    fn from_poly_projection_obligation(
        selcx: &mut SelectionContext<'cx, 'tcx>,
        obligation: &PolyProjectionObligation<'tcx>,
    ) -> Option<Self> {
        let infcx = selcx.infcx;
        // We don't do cross-snapshot caching of obligations with escaping regions,
        // so there's no cache key to use
        obligation.predicate.no_bound_vars().map(|predicate| {
            ProjectionCacheKey::new(
                // We don't attempt to match up with a specific type-variable state
                // from a specific call to `opt_normalize_projection_type` - if
                // there's no precise match, the original cache entry is "stranded"
                // anyway.
                infcx.resolve_vars_if_possible(predicate.projection_term),
                obligation.param_env,
            )
        })
    }
}
