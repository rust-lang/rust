//! Code for projecting associated types out of trait references.

use super::specialization_graph;
use super::translate_substs;
use super::util;
use super::MismatchedProjectionTypes;
use super::Obligation;
use super::ObligationCause;
use super::PredicateObligation;
use super::Selection;
use super::SelectionContext;
use super::SelectionError;
use super::{
    ImplSourceClosureData, ImplSourceFnPointerData, ImplSourceFutureData, ImplSourceGeneratorData,
    ImplSourceUserDefinedData,
};
use super::{Normalized, NormalizedTy, ProjectionCacheEntry, ProjectionCacheKey};

use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use crate::traits::error_reporting::TypeErrCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::select::ProjectionMatchesProjection;
use rustc_data_structures::sso::SsoHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::at::At;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::traits::ImplSourceBuiltinData;
use rustc_middle::traits::select::OverflowError;
use rustc_middle::ty::fold::{ir::TypeFolder, TypeFoldable, TypeSuperFoldable};
use rustc_middle::ty::visit::{MaxUniverse, TypeVisitable};
use rustc_middle::ty::DefIdTree;
use rustc_middle::ty::{self, Term, ToPredicate, Ty, TyCtxt};
use rustc_span::symbol::sym;

use std::collections::BTreeMap;

pub use rustc_middle::traits::Reveal;

pub type PolyProjectionObligation<'tcx> = Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> = Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> = Obligation<'tcx, ty::AliasTy<'tcx>>;

pub(super) struct InProgress;

pub trait NormalizeExt<'tcx> {
    /// Normalize a value using the `AssocTypeNormalizer`.
    ///
    /// This normalization should be used when the type contains inference variables or the
    /// projection may be fallible.
    fn normalize<T: TypeFoldable<'tcx>>(&self, t: T) -> InferOk<'tcx, T>;
}

impl<'tcx> NormalizeExt<'tcx> for At<'_, 'tcx> {
    fn normalize<T: TypeFoldable<'tcx>>(&self, value: T) -> InferOk<'tcx, T> {
        let mut selcx = SelectionContext::new(self.infcx);
        let Normalized { value, obligations } =
            normalize_with_depth(&mut selcx, self.param_env, self.cause.clone(), 0, value);
        InferOk { value, obligations }
    }
}

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

    /// From an "impl" (or a "pseudo-impl" returned by select)
    Select(Selection<'tcx>),

    ImplTraitInTrait(ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>>),
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
                    (_, ParamEnv(..)) => unreachable!(),
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
///     Result<Option<Vec<PredicateObligation<'tcx>>>, InProgress>,
///     MismatchedProjectionTypes<'tcx>,
/// >
/// ```
pub(super) enum ProjectAndUnifyResult<'tcx> {
    /// The projection bound holds subject to the given obligations. If the
    /// projection cannot be normalized because the required trait bound does
    /// not hold, this is returned, with `obligations` being a predicate that
    /// cannot be proven.
    Holds(Vec<PredicateObligation<'tcx>>),
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
pub(super) fn poly_project_and_unify_type<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &PolyProjectionObligation<'tcx>,
) -> ProjectAndUnifyResult<'tcx> {
    let infcx = selcx.infcx;
    let r = infcx.commit_if_ok(|_snapshot| {
        let old_universe = infcx.universe();
        let placeholder_predicate =
            infcx.instantiate_binder_with_placeholders(obligation.predicate);
        let new_universe = infcx.universe();

        let placeholder_obligation = obligation.with(infcx.tcx, placeholder_predicate);
        match project_and_unify_type(selcx, &placeholder_obligation) {
            ProjectAndUnifyResult::MismatchedProjectionTypes(e) => Err(e),
            ProjectAndUnifyResult::Holds(obligations)
                if old_universe != new_universe
                    && selcx.tcx().features().generic_associated_types_extended =>
            {
                // If the `generic_associated_types_extended` feature is active, then we ignore any
                // obligations references lifetimes from any universe greater than or equal to the
                // universe just created. Otherwise, we can end up with something like `for<'a> I: 'a`,
                // which isn't quite what we want. Ideally, we want either an implied
                // `for<'a where I: 'a> I: 'a` or we want to "lazily" check these hold when we
                // substitute concrete regions. There is design work to be done here; until then,
                // however, this allows experimenting potential GAT features without running into
                // well-formedness issues.
                let new_obligations = obligations
                    .into_iter()
                    .filter(|obligation| {
                        let mut visitor = MaxUniverse::new();
                        obligation.predicate.visit_with(&mut visitor);
                        visitor.max_universe() < new_universe
                    })
                    .collect();
                Ok(ProjectAndUnifyResult::Holds(new_obligations))
            }
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
/// See [poly_project_and_unify_type] for an explanation of the return value.
#[instrument(level = "debug", skip(selcx))]
fn project_and_unify_type<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionObligation<'tcx>,
) -> ProjectAndUnifyResult<'tcx> {
    let mut obligations = vec![];

    let infcx = selcx.infcx;
    let normalized = match opt_normalize_projection_type(
        selcx,
        obligation.param_env,
        obligation.predicate.projection_ty,
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

    match infcx.at(&obligation.cause, obligation.param_env).eq(normalized, actual) {
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

/// As `normalize`, but with a custom depth.
pub(crate) fn normalize_with_depth<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: T,
) -> Normalized<'tcx, T>
where
    T: TypeFoldable<'tcx>,
{
    let mut obligations = Vec::new();
    let value = normalize_with_depth_to(selcx, param_env, cause, depth, value, &mut obligations);
    Normalized { value, obligations }
}

#[instrument(level = "info", skip(selcx, param_env, cause, obligations))]
pub(crate) fn normalize_with_depth_to<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: T,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!(obligations.len = obligations.len());
    let mut normalizer = AssocTypeNormalizer::new(selcx, param_env, cause, depth, obligations);
    let result = ensure_sufficient_stack(|| normalizer.fold(value));
    debug!(?result, obligations.len = normalizer.obligations.len());
    debug!(?normalizer.obligations,);
    result
}

#[instrument(level = "info", skip(selcx, param_env, cause, obligations))]
pub(crate) fn try_normalize_with_depth_to<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: T,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!(obligations.len = obligations.len());
    let mut normalizer = AssocTypeNormalizer::new_without_eager_inference_replacement(
        selcx,
        param_env,
        cause,
        depth,
        obligations,
    );
    let result = ensure_sufficient_stack(|| normalizer.fold(value));
    debug!(?result, obligations.len = normalizer.obligations.len());
    debug!(?normalizer.obligations,);
    result
}

pub(crate) fn needs_normalization<'tcx, T: TypeVisitable<'tcx>>(value: &T, reveal: Reveal) -> bool {
    match reveal {
        Reveal::UserFacing => value
            .has_type_flags(ty::TypeFlags::HAS_TY_PROJECTION | ty::TypeFlags::HAS_CT_PROJECTION),
        Reveal::All => value.has_type_flags(
            ty::TypeFlags::HAS_TY_PROJECTION
                | ty::TypeFlags::HAS_TY_OPAQUE
                | ty::TypeFlags::HAS_CT_PROJECTION,
        ),
    }
}

struct AssocTypeNormalizer<'a, 'b, 'tcx> {
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    obligations: &'a mut Vec<PredicateObligation<'tcx>>,
    depth: usize,
    universes: Vec<Option<ty::UniverseIndex>>,
    /// If true, when a projection is unable to be completed, an inference
    /// variable will be created and an obligation registered to project to that
    /// inference variable. Also, constants will be eagerly evaluated.
    eager_inference_replacement: bool,
}

impl<'a, 'b, 'tcx> AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn new(
        selcx: &'a mut SelectionContext<'b, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        depth: usize,
        obligations: &'a mut Vec<PredicateObligation<'tcx>>,
    ) -> AssocTypeNormalizer<'a, 'b, 'tcx> {
        AssocTypeNormalizer {
            selcx,
            param_env,
            cause,
            obligations,
            depth,
            universes: vec![],
            eager_inference_replacement: true,
        }
    }

    fn new_without_eager_inference_replacement(
        selcx: &'a mut SelectionContext<'b, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        depth: usize,
        obligations: &'a mut Vec<PredicateObligation<'tcx>>,
    ) -> AssocTypeNormalizer<'a, 'b, 'tcx> {
        AssocTypeNormalizer {
            selcx,
            param_env,
            cause,
            obligations,
            depth,
            universes: vec![],
            eager_inference_replacement: false,
        }
    }

    fn fold<T: TypeFoldable<'tcx>>(&mut self, value: T) -> T {
        let value = self.selcx.infcx.resolve_vars_if_possible(value);
        debug!(?value);

        assert!(
            !value.has_escaping_bound_vars(),
            "Normalizing {:?} without wrapping in a `Binder`",
            value
        );

        if !needs_normalization(&value, self.param_env.reveal()) {
            value
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'b, 'tcx> TypeFolder<TyCtxt<'tcx>> for AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.selcx.tcx()
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.universes.push(None);
        let t = t.super_fold_with(self);
        self.universes.pop();
        t
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !needs_normalization(&ty, self.param_env.reveal()) {
            return ty;
        }

        // We try to be a little clever here as a performance optimization in
        // cases where there are nested projections under binders.
        // For example:
        // ```
        // for<'a> fn(<T as Foo>::One<'a, Box<dyn Bar<'a, Item=<T as Foo>::Two<'a>>>>)
        // ```
        // We normalize the substs on the projection before the projecting, but
        // if we're naive, we'll
        //   replace bound vars on inner, project inner, replace placeholders on inner,
        //   replace bound vars on outer, project outer, replace placeholders on outer
        //
        // However, if we're a bit more clever, we can replace the bound vars
        // on the entire type before normalizing nested projections, meaning we
        //   replace bound vars on outer, project inner,
        //   project outer, replace placeholders on outer
        //
        // This is possible because the inner `'a` will already be a placeholder
        // when we need to normalize the inner projection
        //
        // On the other hand, this does add a bit of complexity, since we only
        // replace bound vars if the current type is a `Projection` and we need
        // to make sure we don't forget to fold the substs regardless.

        match *ty.kind() {
            // This is really important. While we *can* handle this, this has
            // severe performance implications for large opaque types with
            // late-bound regions. See `issue-88862` benchmark.
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. })
                if !substs.has_escaping_bound_vars() =>
            {
                // Only normalize `impl Trait` outside of type inference, usually in codegen.
                match self.param_env.reveal() {
                    Reveal::UserFacing => ty.super_fold_with(self),

                    Reveal::All => {
                        let recursion_limit = self.interner().recursion_limit();
                        if !recursion_limit.value_within_limit(self.depth) {
                            self.selcx.infcx.err_ctxt().report_overflow_error(
                                &ty,
                                self.cause.span,
                                true,
                                |_| {},
                            );
                        }

                        let substs = substs.fold_with(self);
                        let generic_ty = self.interner().type_of(def_id);
                        let concrete_ty = generic_ty.subst(self.interner(), substs);
                        self.depth += 1;
                        let folded_ty = self.fold_ty(concrete_ty);
                        self.depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Alias(ty::Projection, data) if !data.has_escaping_bound_vars() => {
                // This branch is *mostly* just an optimization: when we don't
                // have escaping bound vars, we don't need to replace them with
                // placeholders (see branch below). *Also*, we know that we can
                // register an obligation to *later* project, since we know
                // there won't be bound vars there.
                let data = data.fold_with(self);
                let normalized_ty = if self.eager_inference_replacement {
                    normalize_projection_type(
                        self.selcx,
                        self.param_env,
                        data,
                        self.cause.clone(),
                        self.depth,
                        &mut self.obligations,
                    )
                } else {
                    opt_normalize_projection_type(
                        self.selcx,
                        self.param_env,
                        data,
                        self.cause.clone(),
                        self.depth,
                        &mut self.obligations,
                    )
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| ty.super_fold_with(self).into())
                };
                debug!(
                    ?self.depth,
                    ?ty,
                    ?normalized_ty,
                    obligations.len = ?self.obligations.len(),
                    "AssocTypeNormalizer: normalized type"
                );
                normalized_ty.ty().unwrap()
            }

            ty::Alias(ty::Projection, data) => {
                // If there are escaping bound vars, we temporarily replace the
                // bound vars with placeholders. Note though, that in the case
                // that we still can't project for whatever reason (e.g. self
                // type isn't known enough), we *can't* register an obligation
                // and return an inference variable (since then that obligation
                // would have bound vars and that's a can of worms). Instead,
                // we just give up and fall back to pretending like we never tried!
                //
                // Note: this isn't necessarily the final approach here; we may
                // want to figure out how to register obligations with escaping vars
                // or handle this some other way.

                let infcx = self.selcx.infcx;
                let (data, mapped_regions, mapped_types, mapped_consts) =
                    BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, data);
                let data = data.fold_with(self);
                let normalized_ty = opt_normalize_projection_type(
                    self.selcx,
                    self.param_env,
                    data,
                    self.cause.clone(),
                    self.depth,
                    &mut self.obligations,
                )
                .ok()
                .flatten()
                .map(|term| term.ty().unwrap())
                .map(|normalized_ty| {
                    PlaceholderReplacer::replace_placeholders(
                        infcx,
                        mapped_regions,
                        mapped_types,
                        mapped_consts,
                        &self.universes,
                        normalized_ty,
                    )
                })
                .unwrap_or_else(|| ty.super_fold_with(self));

                debug!(
                    ?self.depth,
                    ?ty,
                    ?normalized_ty,
                    obligations.len = ?self.obligations.len(),
                    "AssocTypeNormalizer: normalized type"
                );
                normalized_ty
            }

            _ => ty.super_fold_with(self),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_const(&mut self, constant: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let tcx = self.selcx.tcx();
        if tcx.lazy_normalization() || !needs_normalization(&constant, self.param_env.reveal()) {
            constant
        } else {
            let constant = constant.super_fold_with(self);
            debug!(?constant, ?self.param_env);
            with_replaced_escaping_bound_vars(
                self.selcx.infcx,
                &mut self.universes,
                constant,
                |constant| constant.eval(tcx, self.param_env),
            )
        }
    }

    #[inline]
    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.allow_normalization() && needs_normalization(&p, self.param_env.reveal()) {
            p.super_fold_with(self)
        } else {
            p
        }
    }
}

pub struct BoundVarReplacer<'me, 'tcx> {
    infcx: &'me InferCtxt<'tcx>,
    // These three maps track the bound variable that were replaced by placeholders. It might be
    // nice to remove these since we already have the `kind` in the placeholder; we really just need
    // the `var` (but we *could* bring that into scope if we were to track them as we pass them).
    mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
    mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy>,
    mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
    // The current depth relative to *this* folding, *not* the entire normalization. In other words,
    // the depth of binders we've passed here.
    current_index: ty::DebruijnIndex,
    // The `UniverseIndex` of the binding levels above us. These are optional, since we are lazy:
    // we don't actually create a universe until we see a bound var we have to replace.
    universe_indices: &'me mut Vec<Option<ty::UniverseIndex>>,
}

/// Executes `f` on `value` after replacing all escaping bound variables with placeholders
/// and then replaces these placeholders with the original bound variables in the result.
///
/// In most places, bound variables should be replaced right when entering a binder, making
/// this function unnecessary. However, normalization currently does not do that, so we have
/// to do this lazily.
///
/// You should not add any additional uses of this function, at least not without first
/// discussing it with t-types.
///
/// FIXME(@lcnr): We may even consider experimenting with eagerly replacing bound vars during
/// normalization as well, at which point this function will be unnecessary and can be removed.
pub fn with_replaced_escaping_bound_vars<'a, 'tcx, T: TypeFoldable<'tcx>, R: TypeFoldable<'tcx>>(
    infcx: &'a InferCtxt<'tcx>,
    universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
    value: T,
    f: impl FnOnce(T) -> R,
) -> R {
    if value.has_escaping_bound_vars() {
        let (value, mapped_regions, mapped_types, mapped_consts) =
            BoundVarReplacer::replace_bound_vars(infcx, universe_indices, value);
        let result = f(value);
        PlaceholderReplacer::replace_placeholders(
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            result,
        )
    } else {
        f(value)
    }
}

impl<'me, 'tcx> BoundVarReplacer<'me, 'tcx> {
    /// Returns `Some` if we *were* able to replace bound vars. If there are any bound vars that
    /// use a binding level above `universe_indices.len()`, we fail.
    pub fn replace_bound_vars<T: TypeFoldable<'tcx>>(
        infcx: &'me InferCtxt<'tcx>,
        universe_indices: &'me mut Vec<Option<ty::UniverseIndex>>,
        value: T,
    ) -> (
        T,
        BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
        BTreeMap<ty::PlaceholderType, ty::BoundTy>,
        BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
    ) {
        let mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion> = BTreeMap::new();
        let mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy> = BTreeMap::new();
        let mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar> = BTreeMap::new();

        let mut replacer = BoundVarReplacer {
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            current_index: ty::INNERMOST,
            universe_indices,
        };

        let value = value.fold_with(&mut replacer);

        (value, replacer.mapped_regions, replacer.mapped_types, replacer.mapped_consts)
    }

    fn universe_for(&mut self, debruijn: ty::DebruijnIndex) -> ty::UniverseIndex {
        let infcx = self.infcx;
        let index =
            self.universe_indices.len() + self.current_index.as_usize() - debruijn.as_usize() - 1;
        let universe = self.universe_indices[index].unwrap_or_else(|| {
            for i in self.universe_indices.iter_mut().take(index + 1) {
                *i = i.or_else(|| Some(infcx.create_next_universe()))
            }
            self.universe_indices[index].unwrap()
        });
        universe
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for BoundVarReplacer<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!("Bound vars outside of `self.universe_indices`");
            }
            ty::ReLateBound(debruijn, br) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderRegion { universe, name: br.kind };
                self.mapped_regions.insert(p, br);
                self.infcx.tcx.mk_re_placeholder(p)
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(debruijn, _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!("Bound vars outside of `self.universe_indices`");
            }
            ty::Bound(debruijn, bound_ty) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderType { universe, name: bound_ty.kind };
                self.mapped_types.insert(p, bound_ty);
                self.infcx.tcx.mk_placeholder(p)
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(debruijn, _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!("Bound vars outside of `self.universe_indices`");
            }
            ty::ConstKind::Bound(debruijn, bound_const) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderConst { universe, name: bound_const };
                self.mapped_consts.insert(p, bound_const);
                self.infcx.tcx.mk_const(p, ct.ty())
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

/// The inverse of [`BoundVarReplacer`]: replaces placeholders with the bound vars from which they came.
pub struct PlaceholderReplacer<'me, 'tcx> {
    infcx: &'me InferCtxt<'tcx>,
    mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
    mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy>,
    mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
    universe_indices: &'me [Option<ty::UniverseIndex>],
    current_index: ty::DebruijnIndex,
}

impl<'me, 'tcx> PlaceholderReplacer<'me, 'tcx> {
    pub fn replace_placeholders<T: TypeFoldable<'tcx>>(
        infcx: &'me InferCtxt<'tcx>,
        mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
        mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy>,
        mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
        universe_indices: &'me [Option<ty::UniverseIndex>],
        value: T,
    ) -> T {
        let mut replacer = PlaceholderReplacer {
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            current_index: ty::INNERMOST,
        };
        value.fold_with(&mut replacer)
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for PlaceholderReplacer<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        if !t.has_placeholders() && !t.has_infer_regions() {
            return t;
        }
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r0: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let r1 = match *r0 {
            ty::ReVar(_) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_region(self.infcx.tcx, r0),
            _ => r0,
        };

        let r2 = match *r1 {
            ty::RePlaceholder(p) => {
                let replace_var = self.mapped_regions.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                        let db = ty::DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        self.interner().mk_re_late_bound(db, *replace_var)
                    }
                    None => r1,
                }
            }
            _ => r1,
        };

        debug!(?r0, ?r1, ?r2, "fold_region");

        r2
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Placeholder(p) => {
                let replace_var = self.mapped_types.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                        let db = ty::DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        self.interner().mk_bound(db, *replace_var)
                    }
                    None => ty,
                }
            }

            _ if ty.has_placeholders() || ty.has_infer_regions() => ty.super_fold_with(self),
            _ => ty,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Placeholder(p) = ct.kind() {
            let replace_var = self.mapped_consts.get(&p);
            match replace_var {
                Some(replace_var) => {
                    let index = self
                        .universe_indices
                        .iter()
                        .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                        .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                    let db = ty::DebruijnIndex::from_usize(
                        self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                    );
                    self.interner().mk_const(ty::ConstKind::Bound(db, *replace_var), ct.ty())
                }
                None => ct,
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}

/// The guts of `normalize`: normalize a specific projection like `<T
/// as Trait>::Item`. The result is always a type (and possibly
/// additional obligations). If ambiguity arises, which implies that
/// there are unresolved type variables in the projection, we will
/// substitute a fresh type variable `$X` and generate a new
/// obligation `<T as Trait>::Item == $X` for later.
pub fn normalize_projection_type<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> Term<'tcx> {
    opt_normalize_projection_type(
        selcx,
        param_env,
        projection_ty,
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

        selcx.infcx.infer_projection(param_env, projection_ty, cause, depth + 1, obligations).into()
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
fn opt_normalize_projection_type<'a, 'b, 'tcx>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> Result<Option<Term<'tcx>>, InProgress> {
    let infcx = selcx.infcx;
    // Don't use the projection cache in intercrate mode -
    // the `infcx` may be re-used between intercrate in non-intercrate
    // mode, which could lead to using incorrect cache results.
    let use_cache = !selcx.is_intercrate();

    let projection_ty = infcx.resolve_vars_if_possible(projection_ty);
    let cache_key = ProjectionCacheKey::new(projection_ty);

    // FIXME(#20304) For now, I am caching here, which is good, but it
    // means we don't capture the type variables that are created in
    // the case of ambiguity. Which means we may create a large stream
    // of such variables. OTOH, if we move the caching up a level, we
    // would not benefit from caching when proving `T: Trait<U=Foo>`
    // bounds. It might be the case that we want two distinct caches,
    // or else another kind of cache entry.

    let cache_result = if use_cache {
        infcx.inner.borrow_mut().projection_cache().try_start(cache_key)
    } else {
        Ok(())
    };
    match cache_result {
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

            if use_cache {
                infcx.inner.borrow_mut().projection_cache().recur(cache_key);
            }
            return Err(InProgress);
        }
        Err(ProjectionCacheEntry::Recur) => {
            debug!("recur cache");
            return Err(InProgress);
        }
        Err(ProjectionCacheEntry::NormalizedTy { ty, complete: _ }) => {
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
            let result = normalize_to_error(selcx, param_env, projection_ty, cause, depth);
            obligations.extend(result.obligations);
            return Ok(Some(result.value.into()));
        }
    }

    let obligation =
        Obligation::with_depth(selcx.tcx(), cause.clone(), depth, param_env, projection_ty);

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

            let mut result = if projected_term.has_projections() {
                let mut normalizer = AssocTypeNormalizer::new(
                    selcx,
                    param_env,
                    cause,
                    depth + 1,
                    &mut projected_obligations,
                );
                let normalized_ty = normalizer.fold(projected_term);

                Normalized { value: normalized_ty, obligations: projected_obligations }
            } else {
                Normalized { value: projected_term, obligations: projected_obligations }
            };

            let mut deduped: SsoHashSet<_> = Default::default();
            result.obligations.drain_filter(|projected_obligation| {
                if !deduped.insert(projected_obligation.clone()) {
                    return true;
                }
                false
            });

            if use_cache {
                infcx.inner.borrow_mut().projection_cache().insert_term(cache_key, result.clone());
            }
            obligations.extend(result.obligations);
            Ok(Some(result.value))
        }
        Ok(Projected::NoProgress(projected_ty)) => {
            let result = Normalized { value: projected_ty, obligations: vec![] };
            if use_cache {
                infcx.inner.borrow_mut().projection_cache().insert_term(cache_key, result.clone());
            }
            // No need to extend `obligations`.
            Ok(Some(result.value))
        }
        Err(ProjectionError::TooManyCandidates) => {
            debug!("opt_normalize_projection_type: too many candidates");
            if use_cache {
                infcx.inner.borrow_mut().projection_cache().ambiguous(cache_key);
            }
            Ok(None)
        }
        Err(ProjectionError::TraitSelectionError(_)) => {
            debug!("opt_normalize_projection_type: ERROR");
            // if we got an error processing the `T as Trait` part,
            // just return `ty::err` but add the obligation `T :
            // Trait`, which when processed will cause the error to be
            // reported later

            if use_cache {
                infcx.inner.borrow_mut().projection_cache().error(cache_key);
            }
            let result = normalize_to_error(selcx, param_env, projection_ty, cause, depth);
            obligations.extend(result.obligations);
            Ok(Some(result.value.into()))
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
/// Trait>::Foo> to `[type error]` would lead to an obligation of
/// `<MyType<[type error]> as Trait>::Foo`. We are supposed to report
/// an error for this obligation, but we legitimately should not,
/// because it contains `[type error]`. Yuck! (See issue #29857 for
/// one case where this arose.)
fn normalize_to_error<'a, 'tcx>(
    selcx: &mut SelectionContext<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::AliasTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
) -> NormalizedTy<'tcx> {
    let trait_ref = ty::Binder::dummy(projection_ty.trait_ref(selcx.tcx()));
    let trait_obligation = Obligation {
        cause,
        recursion_depth: depth,
        param_env,
        predicate: trait_ref.without_const().to_predicate(selcx.tcx()),
    };
    let tcx = selcx.infcx.tcx;
    let new_value = selcx.infcx.next_ty_var(TypeVariableOrigin {
        kind: TypeVariableOriginKind::NormalizeProjectionType,
        span: tcx.def_span(projection_ty.def_id),
    });
    Normalized { value: new_value, obligations: vec![trait_obligation] }
}

enum Projected<'tcx> {
    Progress(Progress<'tcx>),
    NoProgress(ty::Term<'tcx>),
}

struct Progress<'tcx> {
    term: ty::Term<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> Progress<'tcx> {
    fn error(tcx: TyCtxt<'tcx>) -> Self {
        Progress { term: tcx.ty_error().into(), obligations: vec![] }
    }

    fn with_addl_obligations(mut self, mut obligations: Vec<PredicateObligation<'tcx>>) -> Self {
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
    obligation: &ProjectionTyObligation<'tcx>,
) -> Result<Projected<'tcx>, ProjectionError<'tcx>> {
    if !selcx.tcx().recursion_limit().value_within_limit(obligation.recursion_depth) {
        // This should really be an immediate error, but some existing code
        // relies on being able to recover from this.
        return Err(ProjectionError::TraitSelectionError(SelectionError::Overflow(
            OverflowError::Canonical,
        )));
    }

    if obligation.predicate.references_error() {
        return Ok(Projected::Progress(Progress::error(selcx.tcx())));
    }

    let mut candidates = ProjectionCandidateSet::None;

    assemble_candidate_for_impl_trait_in_trait(selcx, obligation, &mut candidates);

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
        ProjectionCandidateSet::None => Ok(Projected::NoProgress(
            // FIXME(associated_const_generics): this may need to change in the future?
            // need to investigate whether or not this is fine.
            selcx
                .tcx()
                .mk_projection(obligation.predicate.def_id, obligation.predicate.substs)
                .into(),
        )),
        // Error occurred while trying to processing impls.
        ProjectionCandidateSet::Error(e) => Err(ProjectionError::TraitSelectionError(e)),
        // Inherent ambiguity that prevents us from even enumerating the
        // candidates.
        ProjectionCandidateSet::Ambiguous => Err(ProjectionError::TooManyCandidates),
    }
}

/// If the predicate's item is an `ImplTraitPlaceholder`, we do a select on the
/// corresponding trait ref. If this yields an `impl`, then we're able to project
/// to a concrete type, since we have an `impl`'s method  to provide the RPITIT.
fn assemble_candidate_for_impl_trait_in_trait<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    let tcx = selcx.tcx();
    if tcx.def_kind(obligation.predicate.def_id) == DefKind::ImplTraitPlaceholder {
        let trait_fn_def_id = tcx.impl_trait_in_trait_parent(obligation.predicate.def_id);

        let trait_def_id = tcx.parent(trait_fn_def_id);
        let trait_substs =
            obligation.predicate.substs.truncate_to(tcx, tcx.generics_of(trait_def_id));
        // FIXME(named-returns): Binders
        let trait_predicate = ty::Binder::dummy(tcx.mk_trait_ref(trait_def_id, trait_substs));

        let _ = selcx.infcx.commit_if_ok(|_| {
            match selcx.select(&obligation.with(tcx, trait_predicate)) {
                Ok(Some(super::ImplSource::UserDefined(data))) => {
                    candidate_set.push_candidate(ProjectionCandidate::ImplTraitInTrait(data));
                    Ok(())
                }
                Ok(None) => {
                    candidate_set.mark_ambiguous();
                    return Err(());
                }
                Ok(Some(_)) => {
                    // Don't know enough about the impl to provide a useful signature
                    return Err(());
                }
                Err(e) => {
                    debug!(error = ?e, "selection error");
                    candidate_set.mark_error(e);
                    return Err(());
                }
            }
        });
    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
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
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_trait_def(..)");

    let tcx = selcx.tcx();
    // Check whether the self-type is itself a projection.
    // If so, extract what we know from the trait and try to come up with a good answer.
    let bounds = match *obligation.predicate.self_ty().kind() {
        ty::Alias(_, ref data) => tcx.item_bounds(data.def_id).subst(tcx, data.substs),
        ty::Infer(ty::TyVar(_)) => {
            // If the self-type is an inference variable, then it MAY wind up
            // being a projected type, so induce an ambiguity.
            candidate_set.mark_ambiguous();
            return;
        }
        _ => return,
    };

    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionCandidate::TraitDef,
        bounds.iter(),
        true,
    );
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
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_object_ty(..)");

    let tcx = selcx.tcx();

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
        .map(|p| p.with_self_ty(tcx, object_ty).to_predicate(tcx));

    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionCandidate::Object,
        env_predicates,
        false,
    );
}

#[instrument(
    level = "debug",
    skip(selcx, candidate_set, ctor, env_predicates, potentially_unnormalized_candidates)
)]
fn assemble_candidates_from_predicates<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
    ctor: fn(ty::PolyProjectionPredicate<'tcx>) -> ProjectionCandidate<'tcx>,
    env_predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
    potentially_unnormalized_candidates: bool,
) {
    let infcx = selcx.infcx;
    for predicate in env_predicates {
        let bound_predicate = predicate.kind();
        if let ty::PredicateKind::Clause(ty::Clause::Projection(data)) =
            predicate.kind().skip_binder()
        {
            let data = bound_predicate.rebind(data);
            if data.projection_def_id() != obligation.predicate.def_id {
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
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionCandidateSet<'tcx>,
) {
    // Can't assemble candidate from impl for RPITIT
    if selcx.tcx().def_kind(obligation.predicate.def_id) == DefKind::ImplTraitPlaceholder {
        return;
    }

    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let poly_trait_ref = ty::Binder::dummy(obligation.predicate.trait_ref(selcx.tcx()));
    let trait_obligation = obligation.with(selcx.tcx(), poly_trait_ref);
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
            super::ImplSource::Closure(_)
            | super::ImplSource::Generator(_)
            | super::ImplSource::Future(_)
            | super::ImplSource::FnPointer(_)
            | super::ImplSource::TraitAlias(_) => true,
            super::ImplSource::UserDefined(impl_data) => {
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
                let node_item =
                    specialization_graph::assoc_def(selcx.tcx(), impl_data.impl_def_id, obligation.predicate.def_id)
                        .map_err(|ErrorGuaranteed { .. }| ())?;

                if node_item.is_final() {
                    // Non-specializable items are always projectable.
                    true
                } else {
                    // Only reveal a specializable default if we're past type-checking
                    // and the obligation is monomorphic, otherwise passes such as
                    // transmute checking and polymorphic MIR optimizations could
                    // get a result which isn't correct for all monomorphizations.
                    if obligation.param_env.reveal() == Reveal::All {
                        // NOTE(eddyb) inference variables can resolve to parameters, so
                        // assume `poly_trait_ref` isn't monomorphic, if it contains any.
                        let poly_trait_ref = selcx.infcx.resolve_vars_if_possible(poly_trait_ref);
                        !poly_trait_ref.still_further_specializable()
                    } else {
                        debug!(
                            assoc_ty = ?selcx.tcx().def_path_str(node_item.item.def_id),
                            ?obligation.predicate,
                            "assemble_candidates_from_impls: not eligible due to default",
                        );
                        false
                    }
                }
            }
            super::ImplSource::Builtin(..) => {
                // While a builtin impl may be known to exist, the associated type may not yet
                // be known. Any type with multiple potential associated types is therefore
                // not eligible.
                let self_ty = selcx.infcx.shallow_resolve(obligation.predicate.self_ty());

                let lang_items = selcx.tcx().lang_items();
                if lang_items.discriminant_kind_trait() == Some(poly_trait_ref.def_id()) {
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
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::Generator(..)
                        | ty::GeneratorWitness(..)
                        | ty::GeneratorWitnessMIR(..)
                        | ty::Never
                        | ty::Tuple(..)
                        // Integers and floats always have `u8` as their discriminant.
                        | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                         // type parameters, opaques, and unnormalized projections have pointer
                        // metadata if they're known (e.g. by the param_env) to be sized
                        ty::Param(_)
                        | ty::Alias(..)
                        | ty::Bound(..)
                        | ty::Placeholder(..)
                        | ty::Infer(..)
                        | ty::Error(_) => false,
                    }
                } else if lang_items.pointee_trait() == Some(poly_trait_ref.def_id()) {
                    let tail = selcx.tcx().struct_tail_with_normalize(
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
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::Generator(..)
                        | ty::GeneratorWitness(..)
                        | ty::GeneratorWitnessMIR(..)
                        | ty::Never
                        // Extern types have unit metadata, according to RFC 2850
                        | ty::Foreign(_)
                        // If returned by `struct_tail_without_normalization` this is a unit struct
                        // without any fields, or not a struct, and therefore is Sized.
                        | ty::Adt(..)
                        // If returned by `struct_tail_without_normalization` this is the empty tuple.
                        | ty::Tuple(..)
                        // Integers and floats are always Sized, and so have unit type metadata.
                        | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                        // type parameters, opaques, and unnormalized projections have pointer
                        // metadata if they're known (e.g. by the param_env) to be sized
                        ty::Param(_) | ty::Alias(..)
                            if selcx.infcx.predicate_must_hold_modulo_regions(
                                &obligation.with(
                                    selcx.tcx(),
                                    ty::Binder::dummy(
                                        selcx.tcx().at(obligation.cause.span()).mk_trait_ref(LangItem::Sized, [self_ty]),
                                    )
                                    .without_const(),
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
                } else {
                    bug!("unexpected builtin trait with associated type: {poly_trait_ref:?}")
                }
            }
            super::ImplSource::Param(..) => {
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
            super::ImplSource::Object(_) => {
                // Handled by the `Object` projection candidate. See
                // `assemble_candidates_from_object_ty` for an explanation of
                // why we special case object types.
                false
            }
            super::ImplSource::AutoImpl(..)
            | super::ImplSource::TraitUpcasting(_)
            | super::ImplSource::ConstDestruct(_) => {
                // These traits have no associated types.
                selcx.tcx().sess.delay_span_bug(
                    obligation.cause.span,
                    &format!("Cannot project an associated type from `{:?}`", impl_source),
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
    obligation: &ProjectionTyObligation<'tcx>,
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
        ProjectionCandidate::ImplTraitInTrait(data) => {
            confirm_impl_trait_in_trait_candidate(selcx, obligation, data)
        }
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
    obligation: &ProjectionTyObligation<'tcx>,
    impl_source: Selection<'tcx>,
) -> Progress<'tcx> {
    match impl_source {
        super::ImplSource::UserDefined(data) => confirm_impl_candidate(selcx, obligation, data),
        super::ImplSource::Generator(data) => confirm_generator_candidate(selcx, obligation, data),
        super::ImplSource::Future(data) => confirm_future_candidate(selcx, obligation, data),
        super::ImplSource::Closure(data) => confirm_closure_candidate(selcx, obligation, data),
        super::ImplSource::FnPointer(data) => confirm_fn_pointer_candidate(selcx, obligation, data),
        super::ImplSource::Builtin(data) => confirm_builtin_candidate(selcx, obligation, data),
        super::ImplSource::Object(_)
        | super::ImplSource::AutoImpl(..)
        | super::ImplSource::Param(..)
        | super::ImplSource::TraitUpcasting(_)
        | super::ImplSource::TraitAlias(..)
        | super::ImplSource::ConstDestruct(_) => {
            // we don't create Select candidates with this kind of resolution
            span_bug!(
                obligation.cause.span,
                "Cannot project an associated type from `{:?}`",
                impl_source
            )
        }
    }
}

fn confirm_generator_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_source: ImplSourceGeneratorData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let gen_sig = impl_source.substs.as_generator().poly_sig();
    let Normalized { value: gen_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        gen_sig,
    );

    debug!(?obligation, ?gen_sig, ?obligations, "confirm_generator_candidate");

    let tcx = selcx.tcx();

    let gen_def_id = tcx.require_lang_item(LangItem::Generator, None);

    let predicate = super::util::generator_trait_ref_and_outputs(
        tcx,
        gen_def_id,
        obligation.predicate.self_ty(),
        gen_sig,
    )
    .map_bound(|(trait_ref, yield_ty, return_ty)| {
        let name = tcx.associated_item(obligation.predicate.def_id).name;
        let ty = if name == sym::Return {
            return_ty
        } else if name == sym::Yield {
            yield_ty
        } else {
            bug!()
        };

        ty::ProjectionPredicate {
            projection_ty: tcx.mk_alias_ty(obligation.predicate.def_id, trait_ref.substs),
            term: ty.into(),
        }
    });

    confirm_param_env_candidate(selcx, obligation, predicate, false)
        .with_addl_obligations(impl_source.nested)
        .with_addl_obligations(obligations)
}

fn confirm_future_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_source: ImplSourceFutureData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let gen_sig = impl_source.substs.as_generator().poly_sig();
    let Normalized { value: gen_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        gen_sig,
    );

    debug!(?obligation, ?gen_sig, ?obligations, "confirm_future_candidate");

    let tcx = selcx.tcx();
    let fut_def_id = tcx.require_lang_item(LangItem::Future, None);

    let predicate = super::util::future_trait_ref_and_outputs(
        tcx,
        fut_def_id,
        obligation.predicate.self_ty(),
        gen_sig,
    )
    .map_bound(|(trait_ref, return_ty)| {
        debug_assert_eq!(tcx.associated_item(obligation.predicate.def_id).name, sym::Output);

        ty::ProjectionPredicate {
            projection_ty: tcx.mk_alias_ty(obligation.predicate.def_id, trait_ref.substs),
            term: return_ty.into(),
        }
    });

    confirm_param_env_candidate(selcx, obligation, predicate, false)
        .with_addl_obligations(impl_source.nested)
        .with_addl_obligations(obligations)
}

fn confirm_builtin_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    data: ImplSourceBuiltinData<PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let self_ty = obligation.predicate.self_ty();
    let substs = tcx.intern_substs(&[self_ty.into()]);
    let lang_items = tcx.lang_items();
    let item_def_id = obligation.predicate.def_id;
    let trait_def_id = tcx.trait_of_item(item_def_id).unwrap();
    let (term, obligations) = if lang_items.discriminant_kind_trait() == Some(trait_def_id) {
        let discriminant_def_id = tcx.require_lang_item(LangItem::Discriminant, None);
        assert_eq!(discriminant_def_id, item_def_id);

        (self_ty.discriminant_ty(tcx).into(), Vec::new())
    } else if lang_items.pointee_trait() == Some(trait_def_id) {
        let metadata_def_id = tcx.require_lang_item(LangItem::Metadata, None);
        assert_eq!(metadata_def_id, item_def_id);

        let mut obligations = Vec::new();
        let (metadata_ty, check_is_sized) = self_ty.ptr_metadata_ty(tcx, |ty| {
            normalize_with_depth_to(
                selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                ty,
                &mut obligations,
            )
        });
        if check_is_sized {
            let sized_predicate = ty::Binder::dummy(
                tcx.at(obligation.cause.span()).mk_trait_ref(LangItem::Sized, [self_ty]),
            )
            .without_const();
            obligations.push(obligation.with(tcx, sized_predicate));
        }
        (metadata_ty.into(), obligations)
    } else {
        bug!("unexpected builtin trait with associated type: {:?}", obligation.predicate);
    };

    let predicate =
        ty::ProjectionPredicate { projection_ty: tcx.mk_alias_ty(item_def_id, substs), term };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
        .with_addl_obligations(obligations)
        .with_addl_obligations(data.nested)
}

fn confirm_fn_pointer_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_pointer_impl_source: ImplSourceFnPointerData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let fn_type = selcx.infcx.shallow_resolve(fn_pointer_impl_source.fn_ty);
    let sig = fn_type.fn_sig(selcx.tcx());
    let Normalized { value: sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        sig,
    );

    confirm_callable_candidate(selcx, obligation, sig, util::TupleArgumentsFlag::Yes)
        .with_addl_obligations(fn_pointer_impl_source.nested)
        .with_addl_obligations(obligations)
}

fn confirm_closure_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_source: ImplSourceClosureData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let closure_sig = impl_source.substs.as_closure().sig();
    let Normalized { value: closure_sig, obligations } = normalize_with_depth(
        selcx,
        obligation.param_env,
        obligation.cause.clone(),
        obligation.recursion_depth + 1,
        closure_sig,
    );

    debug!(?obligation, ?closure_sig, ?obligations, "confirm_closure_candidate");

    confirm_callable_candidate(selcx, obligation, closure_sig, util::TupleArgumentsFlag::No)
        .with_addl_obligations(impl_source.nested)
        .with_addl_obligations(obligations)
}

fn confirm_callable_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
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
        projection_ty: tcx.mk_alias_ty(fn_once_output_def_id, trait_ref.substs),
        term: ret_type.into(),
    });

    confirm_param_env_candidate(selcx, obligation, predicate, true)
}

fn confirm_param_env_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    poly_cache_entry: ty::PolyProjectionPredicate<'tcx>,
    potentially_unnormalized_candidate: bool,
) -> Progress<'tcx> {
    let infcx = selcx.infcx;
    let cause = &obligation.cause;
    let param_env = obligation.param_env;

    let cache_entry = infcx.instantiate_binder_with_fresh_vars(
        cause.span,
        LateBoundRegionConversionTime::HigherRankedType,
        poly_cache_entry,
    );

    let cache_projection = cache_entry.projection_ty;
    let mut nested_obligations = Vec::new();
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

    match infcx.at(cause, param_env).eq(cache_projection, obligation_projection) {
        Ok(InferOk { value: _, obligations }) => {
            nested_obligations.extend(obligations);
            assoc_ty_own_obligations(selcx, obligation, &mut nested_obligations);
            // FIXME(associated_const_equality): Handle consts here as well? Maybe this progress type should just take
            // a term instead.
            Progress { term: cache_entry.term, obligations: nested_obligations }
        }
        Err(e) => {
            let msg = format!(
                "Failed to unify obligation `{:?}` with poly_projection `{:?}`: {:?}",
                obligation, poly_cache_entry, e,
            );
            debug!("confirm_param_env_candidate: {}", msg);
            let err = infcx.tcx.ty_error_with_message(obligation.cause.span, &msg);
            Progress { term: err.into(), obligations: vec![] }
        }
    }
}

fn confirm_impl_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    impl_impl_source: ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();

    let ImplSourceUserDefinedData { impl_def_id, substs, mut nested } = impl_impl_source;
    let assoc_item_id = obligation.predicate.def_id;
    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();

    let param_env = obligation.param_env;
    let Ok(assoc_ty) = specialization_graph::assoc_def(tcx, impl_def_id, assoc_item_id) else {
        return Progress { term: tcx.ty_error().into(), obligations: nested };
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
        return Progress { term: tcx.ty_error().into(), obligations: nested };
    }
    // If we're trying to normalize `<Vec<u32> as X>::A<S>` using
    //`impl<T> X for Vec<T> { type A<Y> = Box<Y>; }`, then:
    //
    // * `obligation.predicate.substs` is `[Vec<u32>, S]`
    // * `substs` is `[u32]`
    // * `substs` ends up as `[u32, S]`
    let substs = obligation.predicate.substs.rebase_onto(tcx, trait_def_id, substs);
    let substs =
        translate_substs(selcx.infcx, param_env, impl_def_id, substs, assoc_ty.defining_node);
    let ty = tcx.type_of(assoc_ty.item.def_id);
    let is_const = matches!(tcx.def_kind(assoc_ty.item.def_id), DefKind::AssocConst);
    let term: ty::EarlyBinder<ty::Term<'tcx>> = if is_const {
        let identity_substs =
            crate::traits::InternalSubsts::identity_for_item(tcx, assoc_ty.item.def_id);
        let did = ty::WithOptConstParam::unknown(assoc_ty.item.def_id);
        let kind = ty::ConstKind::Unevaluated(ty::UnevaluatedConst::new(did, identity_substs));
        ty.map_bound(|ty| tcx.mk_const(kind, ty).into())
    } else {
        ty.map_bound(|ty| ty.into())
    };
    if !check_substs_compatible(tcx, assoc_ty.item, substs) {
        let err = tcx.ty_error_with_message(
            obligation.cause.span,
            "impl item and trait item have different parameters",
        );
        Progress { term: err.into(), obligations: nested }
    } else {
        assoc_ty_own_obligations(selcx, obligation, &mut nested);
        Progress { term: term.subst(tcx, substs), obligations: nested }
    }
}

// Verify that the trait item and its implementation have compatible substs lists
fn check_substs_compatible<'tcx>(
    tcx: TyCtxt<'tcx>,
    assoc_item: ty::AssocItem,
    substs: ty::SubstsRef<'tcx>,
) -> bool {
    fn check_substs_compatible_inner<'tcx>(
        tcx: TyCtxt<'tcx>,
        generics: &'tcx ty::Generics,
        args: &'tcx [ty::GenericArg<'tcx>],
    ) -> bool {
        if generics.count() != args.len() {
            return false;
        }

        let (parent_args, own_args) = args.split_at(generics.parent_count);

        if let Some(parent) = generics.parent
            && let parent_generics = tcx.generics_of(parent)
            && !check_substs_compatible_inner(tcx, parent_generics, parent_args) {
            return false;
        }

        for (param, arg) in std::iter::zip(&generics.params, own_args) {
            match (&param.kind, arg.unpack()) {
                (ty::GenericParamDefKind::Type { .. }, ty::GenericArgKind::Type(_))
                | (ty::GenericParamDefKind::Lifetime, ty::GenericArgKind::Lifetime(_))
                | (ty::GenericParamDefKind::Const { .. }, ty::GenericArgKind::Const(_)) => {}
                _ => return false,
            }
        }

        true
    }

    let generics = tcx.generics_of(assoc_item.def_id);
    // Chop off any additional substs (RPITIT) substs
    let substs = &substs[0..generics.count().min(substs.len())];
    check_substs_compatible_inner(tcx, generics, substs)
}

fn confirm_impl_trait_in_trait_candidate<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    data: ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();
    let mut obligations = data.nested;

    let trait_fn_def_id = tcx.impl_trait_in_trait_parent(obligation.predicate.def_id);
    let Ok(leaf_def) = specialization_graph::assoc_def(tcx, data.impl_def_id, trait_fn_def_id) else {
        return Progress { term: tcx.ty_error().into(), obligations };
    };
    if !leaf_def.item.defaultness(tcx).has_value() {
        return Progress { term: tcx.ty_error().into(), obligations };
    }

    // Use the default `impl Trait` for the trait, e.g., for a default trait body
    if leaf_def.item.container == ty::AssocItemContainer::TraitContainer {
        return Progress {
            term: tcx.mk_opaque(obligation.predicate.def_id, obligation.predicate.substs).into(),
            obligations,
        };
    }

    // Rebase from {trait}::{fn}::{opaque} to {impl}::{fn}::{opaque},
    // since `data.substs` are the impl substs.
    let impl_fn_substs =
        obligation.predicate.substs.rebase_onto(tcx, tcx.parent(trait_fn_def_id), data.substs);
    let impl_fn_substs = translate_substs(
        selcx.infcx,
        obligation.param_env,
        data.impl_def_id,
        impl_fn_substs,
        leaf_def.defining_node,
    );

    if !check_substs_compatible(tcx, leaf_def.item, impl_fn_substs) {
        let err = tcx.ty_error_with_message(
            obligation.cause.span,
            "impl method and trait method have different parameters",
        );
        return Progress { term: err.into(), obligations };
    }

    let impl_fn_def_id = leaf_def.item.def_id;

    let cause = ObligationCause::new(
        obligation.cause.span,
        obligation.cause.body_id,
        super::ItemObligation(impl_fn_def_id),
    );
    let predicates = normalize_with_depth_to(
        selcx,
        obligation.param_env,
        cause.clone(),
        obligation.recursion_depth + 1,
        tcx.predicates_of(impl_fn_def_id).instantiate(tcx, impl_fn_substs),
        &mut obligations,
    );
    obligations.extend(predicates.into_iter().map(|(pred, span)| {
        Obligation::with_depth(
            tcx,
            ObligationCause::new(
                obligation.cause.span,
                obligation.cause.body_id,
                if span.is_dummy() {
                    super::ItemObligation(impl_fn_def_id)
                } else {
                    super::BindingObligation(impl_fn_def_id, span)
                },
            ),
            obligation.recursion_depth + 1,
            obligation.param_env,
            pred,
        )
    }));

    let ty = normalize_with_depth_to(
        selcx,
        obligation.param_env,
        cause.clone(),
        obligation.recursion_depth + 1,
        tcx.bound_return_position_impl_trait_in_trait_tys(impl_fn_def_id)
            .map_bound(|tys| {
                tys.map_or_else(|_| tcx.ty_error(), |tys| tys[&obligation.predicate.def_id])
            })
            .subst(tcx, impl_fn_substs),
        &mut obligations,
    );

    Progress { term: ty.into(), obligations }
}

// Get obligations corresponding to the predicates from the where-clause of the
// associated type itself.
fn assoc_ty_own_obligations<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    nested: &mut Vec<PredicateObligation<'tcx>>,
) {
    let tcx = selcx.tcx();
    let predicates = tcx
        .predicates_of(obligation.predicate.def_id)
        .instantiate_own(tcx, obligation.predicate.substs);
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
            super::CompareImplItemObligation { .. }
                | super::CheckAssociatedTypeBounds { .. }
                | super::AscribeUserTypeProvePredicate(..)
        ) {
            obligation.cause.clone()
        } else if span.is_dummy() {
            ObligationCause::new(
                obligation.cause.span,
                obligation.cause.body_id,
                super::ItemObligation(obligation.predicate.def_id),
            )
        } else {
            ObligationCause::new(
                obligation.cause.span,
                obligation.cause.body_id,
                super::BindingObligation(obligation.predicate.def_id, span),
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
    fn from_poly_projection_predicate(
        selcx: &mut SelectionContext<'cx, 'tcx>,
        predicate: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<Self>;
}

impl<'cx, 'tcx> ProjectionCacheKeyExt<'cx, 'tcx> for ProjectionCacheKey<'tcx> {
    fn from_poly_projection_predicate(
        selcx: &mut SelectionContext<'cx, 'tcx>,
        predicate: ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<Self> {
        let infcx = selcx.infcx;
        // We don't do cross-snapshot caching of obligations with escaping regions,
        // so there's no cache key to use
        predicate.no_bound_vars().map(|predicate| {
            ProjectionCacheKey::new(
                // We don't attempt to match up with a specific type-variable state
                // from a specific call to `opt_normalize_projection_type` - if
                // there's no precise match, the original cache entry is "stranded"
                // anyway.
                infcx.resolve_vars_if_possible(predicate.projection_ty),
            )
        })
    }
}
