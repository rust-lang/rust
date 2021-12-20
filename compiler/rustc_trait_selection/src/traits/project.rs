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
    ImplSourceClosureData, ImplSourceDiscriminantKindData, ImplSourceFnPointerData,
    ImplSourceGeneratorData, ImplSourcePointeeData, ImplSourceUserDefinedData,
};
use super::{Normalized, NormalizedTy, ProjectionCacheEntry, ProjectionCacheKey};

use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use crate::traits::error_reporting::InferCtxtExt as _;
use rustc_data_structures::sso::SsoHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::ErrorReported;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_span::symbol::sym;

use std::collections::BTreeMap;

pub use rustc_middle::traits::Reveal;

pub type PolyProjectionObligation<'tcx> = Obligation<'tcx, ty::PolyProjectionPredicate<'tcx>>;

pub type ProjectionObligation<'tcx> = Obligation<'tcx, ty::ProjectionPredicate<'tcx>>;

pub type ProjectionTyObligation<'tcx> = Obligation<'tcx, ty::ProjectionTy<'tcx>>;

pub(super) struct InProgress;

/// When attempting to resolve `<T as TraitRef>::Name` ...
#[derive(Debug)]
pub enum ProjectionTyError<'tcx> {
    /// ...we found multiple sources of information and couldn't resolve the ambiguity.
    TooManyCandidates,

    /// ...an error occurred matching `T : TraitRef`
    TraitSelectionError(SelectionError<'tcx>),
}

#[derive(PartialEq, Eq, Debug)]
enum ProjectionTyCandidate<'tcx> {
    /// From a where-clause in the env or object type
    ParamEnv(ty::PolyProjectionPredicate<'tcx>),

    /// From the definition of `Trait` when you have something like <<A as Trait>::B as Trait2>::C
    TraitDef(ty::PolyProjectionPredicate<'tcx>),

    /// Bounds specified on an object type
    Object(ty::PolyProjectionPredicate<'tcx>),

    /// From an "impl" (or a "pseudo-impl" returned by select)
    Select(Selection<'tcx>),
}

enum ProjectionTyCandidateSet<'tcx> {
    None,
    Single(ProjectionTyCandidate<'tcx>),
    Ambiguous,
    Error(SelectionError<'tcx>),
}

impl<'tcx> ProjectionTyCandidateSet<'tcx> {
    fn mark_ambiguous(&mut self) {
        *self = ProjectionTyCandidateSet::Ambiguous;
    }

    fn mark_error(&mut self, err: SelectionError<'tcx>) {
        *self = ProjectionTyCandidateSet::Error(err);
    }

    // Returns true if the push was successful, or false if the candidate
    // was discarded -- this could be because of ambiguity, or because
    // a higher-priority candidate is already there.
    fn push_candidate(&mut self, candidate: ProjectionTyCandidate<'tcx>) -> bool {
        use self::ProjectionTyCandidate::*;
        use self::ProjectionTyCandidateSet::*;

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
                // candidates, we prefer where-clause candidates over impls.  This
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

/// Evaluates constraints of the form:
///
///     for<...> <T as Trait>::U == V
///
/// If successful, this may result in additional obligations. Also returns
/// the projection cache key used to track these additional obligations.
///
/// ## Returns
///
/// - `Err(_)`: the projection can be normalized, but is not equal to the
///   expected type.
/// - `Ok(Err(InProgress))`: this is called recursively while normalizing
///   the same projection.
/// - `Ok(Ok(None))`: The projection cannot be normalized due to ambiguity
///   (resolving some inference variables in the projection may fix this).
/// - `Ok(Ok(Some(obligations)))`: The projection bound holds subject to
///    the given obligations. If the projection cannot be normalized because
///    the required trait bound doesn't hold this returned with `obligations`
///    being a predicate that cannot be proven.
#[instrument(level = "debug", skip(selcx))]
pub(super) fn poly_project_and_unify_type<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &PolyProjectionObligation<'tcx>,
) -> Result<
    Result<Option<Vec<PredicateObligation<'tcx>>>, InProgress>,
    MismatchedProjectionTypes<'tcx>,
> {
    let infcx = selcx.infcx();
    infcx.commit_if_ok(|_snapshot| {
        let placeholder_predicate =
            infcx.replace_bound_vars_with_placeholders(obligation.predicate);

        let placeholder_obligation = obligation.with(placeholder_predicate);
        let result = project_and_unify_type(selcx, &placeholder_obligation)?;
        Ok(result)
    })
}

/// Evaluates constraints of the form:
///
///     <T as Trait>::U == V
///
/// If successful, this may result in additional obligations.
///
/// See [poly_project_and_unify_type] for an explanation of the return value.
fn project_and_unify_type<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionObligation<'tcx>,
) -> Result<
    Result<Option<Vec<PredicateObligation<'tcx>>>, InProgress>,
    MismatchedProjectionTypes<'tcx>,
> {
    debug!(?obligation, "project_and_unify_type");

    let mut obligations = vec![];
    let normalized_ty = match opt_normalize_projection_type(
        selcx,
        obligation.param_env,
        obligation.predicate.projection_ty,
        obligation.cause.clone(),
        obligation.recursion_depth,
        &mut obligations,
    ) {
        Ok(Some(n)) => n,
        Ok(None) => return Ok(Ok(None)),
        Err(InProgress) => return Ok(Err(InProgress)),
    };

    debug!(?normalized_ty, ?obligations, "project_and_unify_type result");

    let infcx = selcx.infcx();
    match infcx
        .at(&obligation.cause, obligation.param_env)
        .eq(normalized_ty, obligation.predicate.ty)
    {
        Ok(InferOk { obligations: inferred_obligations, value: () }) => {
            obligations.extend(inferred_obligations);
            Ok(Ok(Some(obligations)))
        }
        Err(err) => {
            debug!("project_and_unify_type: equating types encountered error {:?}", err);
            Err(MismatchedProjectionTypes { err })
        }
    }
}

/// Normalizes any associated type projections in `value`, replacing
/// them with a fully resolved type where possible. The return value
/// combines the normalized result and any additional obligations that
/// were incurred as result.
pub fn normalize<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    value: T,
) -> Normalized<'tcx, T>
where
    T: TypeFoldable<'tcx>,
{
    let mut obligations = Vec::new();
    let value = normalize_to(selcx, param_env, cause, value, &mut obligations);
    Normalized { value, obligations }
}

pub fn normalize_to<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    value: T,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    normalize_with_depth_to(selcx, param_env, cause, 0, value, obligations)
}

/// As `normalize`, but with a custom depth.
pub fn normalize_with_depth<'a, 'b, 'tcx, T>(
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
pub fn normalize_with_depth_to<'a, 'b, 'tcx, T>(
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

pub(crate) fn needs_normalization<'tcx, T: TypeFoldable<'tcx>>(value: &T, reveal: Reveal) -> bool {
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
}

impl<'a, 'b, 'tcx> AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn new(
        selcx: &'a mut SelectionContext<'b, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        depth: usize,
        obligations: &'a mut Vec<PredicateObligation<'tcx>>,
    ) -> AssocTypeNormalizer<'a, 'b, 'tcx> {
        AssocTypeNormalizer { selcx, param_env, cause, obligations, depth, universes: vec![] }
    }

    fn fold<T: TypeFoldable<'tcx>>(&mut self, value: T) -> T {
        let value = self.selcx.infcx().resolve_vars_if_possible(value);
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

impl<'a, 'b, 'tcx> TypeFolder<'tcx> for AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'tcx> {
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
            ty::Opaque(def_id, substs) if !substs.has_escaping_bound_vars() => {
                // Only normalize `impl Trait` after type-checking, usually in codegen.
                match self.param_env.reveal() {
                    Reveal::UserFacing => ty.super_fold_with(self),

                    Reveal::All => {
                        let recursion_limit = self.tcx().recursion_limit();
                        if !recursion_limit.value_within_limit(self.depth) {
                            let obligation = Obligation::with_depth(
                                self.cause.clone(),
                                recursion_limit.0,
                                self.param_env,
                                ty,
                            );
                            self.selcx.infcx().report_overflow_error(&obligation, true);
                        }

                        let substs = substs.super_fold_with(self);
                        let generic_ty = self.tcx().type_of(def_id);
                        let concrete_ty = generic_ty.subst(self.tcx(), substs);
                        self.depth += 1;
                        let folded_ty = self.fold_ty(concrete_ty);
                        self.depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Projection(data) if !data.has_escaping_bound_vars() => {
                // This branch is *mostly* just an optimization: when we don't
                // have escaping bound vars, we don't need to replace them with
                // placeholders (see branch below). *Also*, we know that we can
                // register an obligation to *later* project, since we know
                // there won't be bound vars there.

                let data = data.super_fold_with(self);
                let normalized_ty = normalize_projection_type(
                    self.selcx,
                    self.param_env,
                    data,
                    self.cause.clone(),
                    self.depth,
                    &mut self.obligations,
                );
                debug!(
                    ?self.depth,
                    ?ty,
                    ?normalized_ty,
                    obligations.len = ?self.obligations.len(),
                    "AssocTypeNormalizer: normalized type"
                );
                normalized_ty
            }

            ty::Projection(data) => {
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

                let infcx = self.selcx.infcx();
                let (data, mapped_regions, mapped_types, mapped_consts) =
                    BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, data);
                let data = data.super_fold_with(self);
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

    fn fold_const(&mut self, constant: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if self.selcx.tcx().lazy_normalization() {
            constant
        } else {
            let constant = constant.super_fold_with(self);
            constant.eval(self.selcx.tcx(), self.param_env)
        }
    }
}

pub struct BoundVarReplacer<'me, 'tcx> {
    infcx: &'me InferCtxt<'me, 'tcx>,
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

impl<'me, 'tcx> BoundVarReplacer<'me, 'tcx> {
    /// Returns `Some` if we *were* able to replace bound vars. If there are any bound vars that
    /// use a binding level above `universe_indices.len()`, we fail.
    pub fn replace_bound_vars<T: TypeFoldable<'tcx>>(
        infcx: &'me InferCtxt<'me, 'tcx>,
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

        let value = value.super_fold_with(&mut replacer);

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

impl<'tcx> TypeFolder<'tcx> for BoundVarReplacer<'_, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
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
                self.infcx.tcx.mk_region(ty::RePlaceholder(p))
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
                let p = ty::PlaceholderType { universe, name: bound_ty.var };
                self.mapped_types.insert(p, bound_ty);
                self.infcx.tcx.mk_ty(ty::Placeholder(p))
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        match *ct {
            ty::Const { val: ty::ConstKind::Bound(debruijn, _), ty: _ }
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!("Bound vars outside of `self.universe_indices`");
            }
            ty::Const { val: ty::ConstKind::Bound(debruijn, bound_const), ty }
                if debruijn >= self.current_index =>
            {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderConst {
                    universe,
                    name: ty::BoundConst { var: bound_const, ty },
                };
                self.mapped_consts.insert(p, bound_const);
                self.infcx.tcx.mk_const(ty::Const { val: ty::ConstKind::Placeholder(p), ty })
            }
            _ if ct.has_vars_bound_at_or_above(self.current_index) => ct.super_fold_with(self),
            _ => ct,
        }
    }
}

// The inverse of `BoundVarReplacer`: replaces placeholders with the bound vars from which they came.
pub struct PlaceholderReplacer<'me, 'tcx> {
    infcx: &'me InferCtxt<'me, 'tcx>,
    mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
    mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy>,
    mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
    universe_indices: &'me Vec<Option<ty::UniverseIndex>>,
    current_index: ty::DebruijnIndex,
}

impl<'me, 'tcx> PlaceholderReplacer<'me, 'tcx> {
    pub fn replace_placeholders<T: TypeFoldable<'tcx>>(
        infcx: &'me InferCtxt<'me, 'tcx>,
        mapped_regions: BTreeMap<ty::PlaceholderRegion, ty::BoundRegion>,
        mapped_types: BTreeMap<ty::PlaceholderType, ty::BoundTy>,
        mapped_consts: BTreeMap<ty::PlaceholderConst<'tcx>, ty::BoundVar>,
        universe_indices: &'me Vec<Option<ty::UniverseIndex>>,
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
        value.super_fold_with(&mut replacer)
    }
}

impl<'tcx> TypeFolder<'tcx> for PlaceholderReplacer<'_, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
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
        let r1 = match r0 {
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
                        self.tcx().mk_region(ty::ReLateBound(db, *replace_var))
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
                        self.tcx().mk_ty(ty::Bound(db, *replace_var))
                    }
                    None => ty,
                }
            }

            _ if ty.has_placeholders() || ty.has_infer_regions() => ty.super_fold_with(self),
            _ => ty,
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if let ty::Const { val: ty::ConstKind::Placeholder(p), ty } = *ct {
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
                    self.tcx()
                        .mk_const(ty::Const { val: ty::ConstKind::Bound(db, *replace_var), ty })
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
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> Ty<'tcx> {
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

        selcx.infcx().infer_projection(param_env, projection_ty, cause, depth + 1, obligations)
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
    projection_ty: ty::ProjectionTy<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    obligations: &mut Vec<PredicateObligation<'tcx>>,
) -> Result<Option<Ty<'tcx>>, InProgress> {
    let infcx = selcx.infcx();
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
            // bootstrapping.  That is, imagine an environment with a
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
            return Ok(Some(result.value));
        }
    }

    let obligation = Obligation::with_depth(cause.clone(), depth, param_env, projection_ty);

    match project_type(selcx, &obligation) {
        Ok(ProjectedTy::Progress(Progress {
            ty: projected_ty,
            obligations: mut projected_obligations,
        })) => {
            // if projection succeeded, then what we get out of this
            // is also non-normalized (consider: it was derived from
            // an impl, where-clause etc) and hence we must
            // re-normalize it

            let projected_ty = selcx.infcx().resolve_vars_if_possible(projected_ty);
            debug!(?projected_ty, ?depth, ?projected_obligations);

            let mut result = if projected_ty.has_projections() {
                let mut normalizer = AssocTypeNormalizer::new(
                    selcx,
                    param_env,
                    cause,
                    depth + 1,
                    &mut projected_obligations,
                );
                let normalized_ty = normalizer.fold(projected_ty);

                debug!(?normalized_ty, ?depth);

                Normalized { value: normalized_ty, obligations: projected_obligations }
            } else {
                Normalized { value: projected_ty, obligations: projected_obligations }
            };

            let mut deduped: SsoHashSet<_> = Default::default();
            result.obligations.drain_filter(|projected_obligation| {
                if !deduped.insert(projected_obligation.clone()) {
                    return true;
                }
                false
            });

            if use_cache {
                infcx.inner.borrow_mut().projection_cache().insert_ty(cache_key, result.clone());
            }
            obligations.extend(result.obligations);
            Ok(Some(result.value))
        }
        Ok(ProjectedTy::NoProgress(projected_ty)) => {
            debug!(?projected_ty, "opt_normalize_projection_type: no progress");
            let result = Normalized { value: projected_ty, obligations: vec![] };
            if use_cache {
                infcx.inner.borrow_mut().projection_cache().insert_ty(cache_key, result.clone());
            }
            // No need to extend `obligations`.
            Ok(Some(result.value))
        }
        Err(ProjectionTyError::TooManyCandidates) => {
            debug!("opt_normalize_projection_type: too many candidates");
            if use_cache {
                infcx.inner.borrow_mut().projection_cache().ambiguous(cache_key);
            }
            Ok(None)
        }
        Err(ProjectionTyError::TraitSelectionError(_)) => {
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
/// Trait>::Foo> to `[type error]` would lead to an obligation of
/// `<MyType<[type error]> as Trait>::Foo`. We are supposed to report
/// an error for this obligation, but we legitimately should not,
/// because it contains `[type error]`. Yuck! (See issue #29857 for
/// one case where this arose.)
fn normalize_to_error<'a, 'tcx>(
    selcx: &mut SelectionContext<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    projection_ty: ty::ProjectionTy<'tcx>,
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
    let tcx = selcx.infcx().tcx;
    let def_id = projection_ty.item_def_id;
    let new_value = selcx.infcx().next_ty_var(TypeVariableOrigin {
        kind: TypeVariableOriginKind::NormalizeProjectionType,
        span: tcx.def_span(def_id),
    });
    Normalized { value: new_value, obligations: vec![trait_obligation] }
}

enum ProjectedTy<'tcx> {
    Progress(Progress<'tcx>),
    NoProgress(Ty<'tcx>),
}

struct Progress<'tcx> {
    ty: Ty<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> Progress<'tcx> {
    fn error(tcx: TyCtxt<'tcx>) -> Self {
        Progress { ty: tcx.ty_error(), obligations: vec![] }
    }

    fn with_addl_obligations(mut self, mut obligations: Vec<PredicateObligation<'tcx>>) -> Self {
        debug!(
            self.obligations.len = ?self.obligations.len(),
            obligations.len = obligations.len(),
            "with_addl_obligations"
        );

        debug!(?self.obligations, ?obligations, "with_addl_obligations");

        self.obligations.append(&mut obligations);
        self
    }
}

/// Computes the result of a projection type (if we can).
///
/// IMPORTANT:
/// - `obligation` must be fully normalized
#[tracing::instrument(level = "info", skip(selcx))]
fn project_type<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
) -> Result<ProjectedTy<'tcx>, ProjectionTyError<'tcx>> {
    if !selcx.tcx().recursion_limit().value_within_limit(obligation.recursion_depth) {
        debug!("project: overflow!");
        // This should really be an immediate error, but some existing code
        // relies on being able to recover from this.
        return Err(ProjectionTyError::TraitSelectionError(SelectionError::Overflow));
    }

    if obligation.predicate.references_error() {
        return Ok(ProjectedTy::Progress(Progress::error(selcx.tcx())));
    }

    let mut candidates = ProjectionTyCandidateSet::None;

    // Make sure that the following procedures are kept in order. ParamEnv
    // needs to be first because it has highest priority, and Select checks
    // the return value of push_candidate which assumes it's ran at last.
    assemble_candidates_from_param_env(selcx, obligation, &mut candidates);

    assemble_candidates_from_trait_def(selcx, obligation, &mut candidates);

    assemble_candidates_from_object_ty(selcx, obligation, &mut candidates);

    if let ProjectionTyCandidateSet::Single(ProjectionTyCandidate::Object(_)) = candidates {
        // Avoid normalization cycle from selection (see
        // `assemble_candidates_from_object_ty`).
        // FIXME(lazy_normalization): Lazy normalization should save us from
        // having to special case this.
    } else {
        assemble_candidates_from_impls(selcx, obligation, &mut candidates);
    };

    match candidates {
        ProjectionTyCandidateSet::Single(candidate) => {
            Ok(ProjectedTy::Progress(confirm_candidate(selcx, obligation, candidate)))
        }
        ProjectionTyCandidateSet::None => Ok(ProjectedTy::NoProgress(
            selcx
                .tcx()
                .mk_projection(obligation.predicate.item_def_id, obligation.predicate.substs),
        )),
        // Error occurred while trying to processing impls.
        ProjectionTyCandidateSet::Error(e) => Err(ProjectionTyError::TraitSelectionError(e)),
        // Inherent ambiguity that prevents us from even enumerating the
        // candidates.
        ProjectionTyCandidateSet::Ambiguous => Err(ProjectionTyError::TooManyCandidates),
    }
}

/// The first thing we have to do is scan through the parameter
/// environment to see whether there are any projection predicates
/// there that can answer this question.
fn assemble_candidates_from_param_env<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_param_env(..)");
    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionTyCandidate::ParamEnv,
        obligation.param_env.caller_bounds().iter(),
        false,
    );
}

/// In the case of a nested projection like <<A as Foo>::FooT as Bar>::BarT, we may find
/// that the definition of `Foo` has some clues:
///
/// ```
/// trait Foo {
///     type FooT : Bar<BarT=i32>
/// }
/// ```
///
/// Here, for example, we could conclude that the result is `i32`.
fn assemble_candidates_from_trait_def<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_trait_def(..)");

    let tcx = selcx.tcx();
    // Check whether the self-type is itself a projection.
    // If so, extract what we know from the trait and try to come up with a good answer.
    let bounds = match *obligation.predicate.self_ty().kind() {
        ty::Projection(ref data) => tcx.item_bounds(data.item_def_id).subst(tcx, data.substs),
        ty::Opaque(def_id, substs) => tcx.item_bounds(def_id).subst(tcx, substs),
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
        ProjectionTyCandidate::TraitDef,
        bounds.iter(),
        true,
    )
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
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_object_ty(..)");

    let tcx = selcx.tcx();

    let self_ty = obligation.predicate.self_ty();
    let object_ty = selcx.infcx().shallow_resolve(self_ty);
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
        .filter(|bound| bound.item_def_id() == obligation.predicate.item_def_id)
        .map(|p| p.with_self_ty(tcx, object_ty).to_predicate(tcx));

    assemble_candidates_from_predicates(
        selcx,
        obligation,
        candidate_set,
        ProjectionTyCandidate::Object,
        env_predicates,
        false,
    );
}

fn assemble_candidates_from_predicates<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
    ctor: fn(ty::PolyProjectionPredicate<'tcx>) -> ProjectionTyCandidate<'tcx>,
    env_predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
    potentially_unnormalized_candidates: bool,
) {
    debug!(?obligation, "assemble_candidates_from_predicates");

    let infcx = selcx.infcx();
    for predicate in env_predicates {
        debug!(?predicate);
        let bound_predicate = predicate.kind();
        if let ty::PredicateKind::Projection(data) = predicate.kind().skip_binder() {
            let data = bound_predicate.rebind(data);
            let same_def_id = data.projection_def_id() == obligation.predicate.item_def_id;

            let is_match = same_def_id
                && infcx.probe(|_| {
                    selcx.match_projection_projections(
                        obligation,
                        data,
                        potentially_unnormalized_candidates,
                    )
                });

            debug!(?data, ?is_match, ?same_def_id);

            if is_match {
                candidate_set.push_candidate(ctor(data));

                if potentially_unnormalized_candidates
                    && !obligation.predicate.has_infer_types_or_consts()
                {
                    // HACK: Pick the first trait def candidate for a fully
                    // inferred predicate. This is to allow duplicates that
                    // differ only in normalization.
                    return;
                }
            }
        }
    }
}

fn assemble_candidates_from_impls<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    candidate_set: &mut ProjectionTyCandidateSet<'tcx>,
) {
    debug!("assemble_candidates_from_impls");

    // If we are resolving `<T as TraitRef<...>>::Item == Type`,
    // start out by selecting the predicate `T as TraitRef<...>`:
    let poly_trait_ref = ty::Binder::dummy(obligation.predicate.trait_ref(selcx.tcx()));
    let trait_obligation = obligation.with(poly_trait_ref.to_poly_trait_predicate());
    let _ = selcx.infcx().commit_if_ok(|_| {
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
            | super::ImplSource::FnPointer(_)
            | super::ImplSource::TraitAlias(_) => {
                debug!(?impl_source);
                true
            }
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
                    assoc_ty_def(selcx, impl_data.impl_def_id, obligation.predicate.item_def_id)
                        .map_err(|ErrorReported| ())?;

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
                        let poly_trait_ref = selcx.infcx().resolve_vars_if_possible(poly_trait_ref);
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
            super::ImplSource::DiscriminantKind(..) => {
                // While `DiscriminantKind` is automatically implemented for every type,
                // the concrete discriminant may not be known yet.
                //
                // Any type with multiple potential discriminant types is therefore not eligible.
                let self_ty = selcx.infcx().shallow_resolve(obligation.predicate.self_ty());

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
                    | ty::Never
                    | ty::Tuple(..)
                    // Integers and floats always have `u8` as their discriminant.
                    | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                    ty::Projection(..)
                    | ty::Opaque(..)
                    | ty::Param(..)
                    | ty::Bound(..)
                    | ty::Placeholder(..)
                    | ty::Infer(..)
                    | ty::Error(_) => false,
                }
            }
            super::ImplSource::Pointee(..) => {
                // While `Pointee` is automatically implemented for every type,
                // the concrete metadata type may not be known yet.
                //
                // Any type with multiple potential metadata types is therefore not eligible.
                let self_ty = selcx.infcx().shallow_resolve(obligation.predicate.self_ty());

                // FIXME: should this normalize?
                let tail = selcx.tcx().struct_tail_without_normalization(self_ty);
                match tail.kind() {
                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
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
                    | ty::Never
                    // If returned by `struct_tail_without_normalization` this is a unit struct
                    // without any fields, or not a struct, and therefore is Sized.
                    | ty::Adt(..)
                    // If returned by `struct_tail_without_normalization` this is the empty tuple.
                    | ty::Tuple(..)
                    // Integers and floats are always Sized, and so have unit type metadata.
                    | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(..)) => true,

                    ty::Projection(..)
                    | ty::Opaque(..)
                    | ty::Param(..)
                    | ty::Bound(..)
                    | ty::Placeholder(..)
                    | ty::Infer(..)
                    | ty::Error(_) => false,
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
                // Doesn't the `T : Sometrait<Foo=usize>` predicate help
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
            | super::ImplSource::Builtin(..)
            | super::ImplSource::TraitUpcasting(_)
            | super::ImplSource::ConstDrop(_) => {
                // These traits have no associated types.
                selcx.tcx().sess.delay_span_bug(
                    obligation.cause.span,
                    &format!("Cannot project an associated type from `{:?}`", impl_source),
                );
                return Err(());
            }
        };

        if eligible {
            if candidate_set.push_candidate(ProjectionTyCandidate::Select(impl_source)) {
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
    candidate: ProjectionTyCandidate<'tcx>,
) -> Progress<'tcx> {
    debug!(?obligation, ?candidate, "confirm_candidate");
    let mut progress = match candidate {
        ProjectionTyCandidate::ParamEnv(poly_projection)
        | ProjectionTyCandidate::Object(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection, false)
        }

        ProjectionTyCandidate::TraitDef(poly_projection) => {
            confirm_param_env_candidate(selcx, obligation, poly_projection, true)
        }

        ProjectionTyCandidate::Select(impl_source) => {
            confirm_select_candidate(selcx, obligation, impl_source)
        }
    };
    // When checking for cycle during evaluation, we compare predicates with
    // "syntactic" equality. Since normalization generally introduces a type
    // with new region variables, we need to resolve them to existing variables
    // when possible for this to work. See `auto-trait-projection-recursion.rs`
    // for a case where this matters.
    if progress.ty.has_infer_regions() {
        progress.ty = OpportunisticRegionResolver::new(selcx.infcx()).fold_ty(progress.ty);
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
        super::ImplSource::Closure(data) => confirm_closure_candidate(selcx, obligation, data),
        super::ImplSource::FnPointer(data) => confirm_fn_pointer_candidate(selcx, obligation, data),
        super::ImplSource::DiscriminantKind(data) => {
            confirm_discriminant_kind_candidate(selcx, obligation, data)
        }
        super::ImplSource::Pointee(data) => confirm_pointee_candidate(selcx, obligation, data),
        super::ImplSource::Object(_)
        | super::ImplSource::AutoImpl(..)
        | super::ImplSource::Param(..)
        | super::ImplSource::Builtin(..)
        | super::ImplSource::TraitUpcasting(_)
        | super::ImplSource::TraitAlias(..)
        | super::ImplSource::ConstDrop(_) => {
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
        let name = tcx.associated_item(obligation.predicate.item_def_id).ident.name;
        let ty = if name == sym::Return {
            return_ty
        } else if name == sym::Yield {
            yield_ty
        } else {
            bug!()
        };

        ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                substs: trait_ref.substs,
                item_def_id: obligation.predicate.item_def_id,
            },
            ty,
        }
    });

    confirm_param_env_candidate(selcx, obligation, predicate, false)
        .with_addl_obligations(impl_source.nested)
        .with_addl_obligations(obligations)
}

fn confirm_discriminant_kind_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    _: ImplSourceDiscriminantKindData,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();

    let self_ty = selcx.infcx().shallow_resolve(obligation.predicate.self_ty());
    // We get here from `poly_project_and_unify_type` which replaces bound vars
    // with placeholders
    debug_assert!(!self_ty.has_escaping_bound_vars());
    let substs = tcx.mk_substs([self_ty.into()].iter());

    let discriminant_def_id = tcx.require_lang_item(LangItem::Discriminant, None);

    let predicate = ty::ProjectionPredicate {
        projection_ty: ty::ProjectionTy { substs, item_def_id: discriminant_def_id },
        ty: self_ty.discriminant_ty(tcx),
    };

    // We get here from `poly_project_and_unify_type` which replaces bound vars
    // with placeholders, so dummy is okay here.
    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
}

fn confirm_pointee_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    _: ImplSourcePointeeData,
) -> Progress<'tcx> {
    let tcx = selcx.tcx();

    let self_ty = selcx.infcx().shallow_resolve(obligation.predicate.self_ty());
    let substs = tcx.mk_substs([self_ty.into()].iter());

    let metadata_def_id = tcx.require_lang_item(LangItem::Metadata, None);

    let predicate = ty::ProjectionPredicate {
        projection_ty: ty::ProjectionTy { substs, item_def_id: metadata_def_id },
        ty: self_ty.ptr_metadata_ty(tcx),
    };

    confirm_param_env_candidate(selcx, obligation, ty::Binder::dummy(predicate), false)
}

fn confirm_fn_pointer_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    fn_pointer_impl_source: ImplSourceFnPointerData<'tcx, PredicateObligation<'tcx>>,
) -> Progress<'tcx> {
    let fn_type = selcx.infcx().shallow_resolve(fn_pointer_impl_source.fn_ty);
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
        projection_ty: ty::ProjectionTy {
            substs: trait_ref.substs,
            item_def_id: fn_once_output_def_id,
        },
        ty: ret_type,
    });

    confirm_param_env_candidate(selcx, obligation, predicate, true)
}

fn confirm_param_env_candidate<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    poly_cache_entry: ty::PolyProjectionPredicate<'tcx>,
    potentially_unnormalized_candidate: bool,
) -> Progress<'tcx> {
    let infcx = selcx.infcx();
    let cause = &obligation.cause;
    let param_env = obligation.param_env;

    let (cache_entry, _) = infcx.replace_bound_vars_with_fresh_vars(
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
            Progress { ty: cache_entry.ty, obligations: nested_obligations }
        }
        Err(e) => {
            let msg = format!(
                "Failed to unify obligation `{:?}` with poly_projection `{:?}`: {:?}",
                obligation, poly_cache_entry, e,
            );
            debug!("confirm_param_env_candidate: {}", msg);
            let err = infcx.tcx.ty_error_with_message(obligation.cause.span, &msg);
            Progress { ty: err, obligations: vec![] }
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
    let assoc_item_id = obligation.predicate.item_def_id;
    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();

    let param_env = obligation.param_env;
    let assoc_ty = match assoc_ty_def(selcx, impl_def_id, assoc_item_id) {
        Ok(assoc_ty) => assoc_ty,
        Err(ErrorReported) => return Progress { ty: tcx.ty_error(), obligations: nested },
    };

    if !assoc_ty.item.defaultness.has_value() {
        // This means that the impl is missing a definition for the
        // associated type. This error will be reported by the type
        // checker method `check_impl_items_against_trait`, so here we
        // just return Error.
        debug!(
            "confirm_impl_candidate: no associated type {:?} for {:?}",
            assoc_ty.item.ident, obligation.predicate
        );
        return Progress { ty: tcx.ty_error(), obligations: nested };
    }
    // If we're trying to normalize `<Vec<u32> as X>::A<S>` using
    //`impl<T> X for Vec<T> { type A<Y> = Box<Y>; }`, then:
    //
    // * `obligation.predicate.substs` is `[Vec<u32>, S]`
    // * `substs` is `[u32]`
    // * `substs` ends up as `[u32, S]`
    let substs = obligation.predicate.substs.rebase_onto(tcx, trait_def_id, substs);
    let substs =
        translate_substs(selcx.infcx(), param_env, impl_def_id, substs, assoc_ty.defining_node);
    let ty = tcx.type_of(assoc_ty.item.def_id);
    if substs.len() != tcx.generics_of(assoc_ty.item.def_id).count() {
        let err = tcx.ty_error_with_message(
            obligation.cause.span,
            "impl item and trait item have different parameter counts",
        );
        Progress { ty: err, obligations: nested }
    } else {
        assoc_ty_own_obligations(selcx, obligation, &mut nested);
        Progress { ty: ty.subst(tcx, substs), obligations: nested }
    }
}

// Get obligations corresponding to the predicates from the where-clause of the
// associated type itself.
// Note: `feature(generic_associated_types)` is required to write such
// predicates, even for non-generic associcated types.
fn assoc_ty_own_obligations<'cx, 'tcx>(
    selcx: &mut SelectionContext<'cx, 'tcx>,
    obligation: &ProjectionTyObligation<'tcx>,
    nested: &mut Vec<PredicateObligation<'tcx>>,
) {
    let tcx = selcx.tcx();
    for predicate in tcx
        .predicates_of(obligation.predicate.item_def_id)
        .instantiate_own(tcx, obligation.predicate.substs)
        .predicates
    {
        let normalized = normalize_with_depth_to(
            selcx,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            predicate,
            nested,
        );
        nested.push(Obligation::with_depth(
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            obligation.param_env,
            normalized,
        ));
    }
}

/// Locate the definition of an associated type in the specialization hierarchy,
/// starting from the given impl.
///
/// Based on the "projection mode", this lookup may in fact only examine the
/// topmost impl. See the comments for `Reveal` for more details.
fn assoc_ty_def(
    selcx: &SelectionContext<'_, '_>,
    impl_def_id: DefId,
    assoc_ty_def_id: DefId,
) -> Result<specialization_graph::LeafDef, ErrorReported> {
    let tcx = selcx.tcx();
    let assoc_ty_name = tcx.associated_item(assoc_ty_def_id).ident;
    let trait_def_id = tcx.impl_trait_ref(impl_def_id).unwrap().def_id;
    let trait_def = tcx.trait_def(trait_def_id);

    // This function may be called while we are still building the
    // specialization graph that is queried below (via TraitDef::ancestors()),
    // so, in order to avoid unnecessary infinite recursion, we manually look
    // for the associated item at the given impl.
    // If there is no such item in that impl, this function will fail with a
    // cycle error if the specialization graph is currently being built.
    let impl_node = specialization_graph::Node::Impl(impl_def_id);
    for item in impl_node.items(tcx) {
        if matches!(item.kind, ty::AssocKind::Type)
            && tcx.hygienic_eq(item.ident, assoc_ty_name, trait_def_id)
        {
            return Ok(specialization_graph::LeafDef {
                item: *item,
                defining_node: impl_node,
                finalizing_node: if item.defaultness.is_default() { None } else { Some(impl_node) },
            });
        }
    }

    let ancestors = trait_def.ancestors(tcx, impl_def_id)?;
    if let Some(assoc_item) = ancestors.leaf_def(tcx, assoc_ty_name, ty::AssocKind::Type) {
        Ok(assoc_item)
    } else {
        // This is saying that neither the trait nor
        // the impl contain a definition for this
        // associated type.  Normally this situation
        // could only arise through a compiler bug --
        // if the user wrote a bad item name, it
        // should have failed in astconv.
        bug!("No associated type `{}` for {}", assoc_ty_name, tcx.def_path_str(impl_def_id))
    }
}

crate trait ProjectionCacheKeyExt<'cx, 'tcx>: Sized {
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
        let infcx = selcx.infcx();
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
