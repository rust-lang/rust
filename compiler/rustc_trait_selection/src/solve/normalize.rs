use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::solve::Goal;
use rustc_infer::traits::{
    FromSolverError, Normalized, Obligation, PredicateObligation, PredicateObligations,
    TraitEngine, TraitErrors,
};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, Binder, Flags, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
    UniverseIndex, Unnormalized,
};
use rustc_next_trait_solver::normalize::{NormalizationFolder, NormalizationWasAmbiguous};
use rustc_next_trait_solver::solve::SolverDelegateEvalExt;
use thin_vec::{ThinVec, thin_vec};

use super::{FulfillmentCtxt, NextSolverError};
use crate::solve::{Certainty, SolverDelegate};
use crate::traits::{BoundVarReplacer, ScrubbedTraitError};

/// see `normalize_with_universes`.
pub fn normalize<'tcx, T>(
    infcx: &InferCtxt<'tcx>,
    cause: &ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: Unnormalized<'tcx, T>,
) -> Normalized<'tcx, T>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    match normalize_with_universes(infcx, value.clone(), vec![], param_env, cause) {
        Ok(normalized) => normalized,
        Err(_) => {
            let mut replacer = ReplaceAliasWithInfer {
                infcx,
                cause,
                param_env,
                obligations: Default::default(),
                universes: vec![],
            };
            let value = infcx.deeply_resolve_ignoring_regions(value.skip_normalization());
            let value = value.fold_with(&mut replacer);
            Normalized { value, obligations: replacer.obligations }
        }
    }
}

/// Like `deeply_normalize`, but we handle ambiguity and inference variables in this routine.
/// The behavior should be same as the old solver.
/// On error, return the failed obligation.
/// For ambiguity, we have two cases:
///   - has_escaping_bound_vars: return the original alias.
///   - otherwise: return the normalized result. It can be (partially) inferred
///     even if the evaluation result is ambiguous.
fn normalize_with_universes<'tcx, T>(
    infcx: &InferCtxt<'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &ObligationCause<'tcx>,
) -> Result<Normalized<'tcx, T>, PredicateObligation<'tcx>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    let value = value.skip_normalization();
    let value = infcx.deeply_resolve_ignoring_regions(value);

    if !infcx.tcx.renormalize_rigid_aliases() && !value.has_non_rigid_aliases() {
        return Ok(Normalized { value, obligations: Default::default() });
    }

    let mut stalled_goals = vec![];
    let mut folder = NormalizationFolder::new(infcx, universes, |alias_term| {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        let infer_term = delegate.next_term_var_of_alias_kind(alias_term, cause.span);
        let predicate = ty::ProjectionClause { projection_term: alias_term, term: infer_term };
        let goal = Goal::new(infcx.tcx, param_env, predicate);
        let result = match delegate.evaluate_root_goal(goal, cause.span, None) {
            Ok(result) => result,
            Err(_) => {
                return Err(Obligation::new(
                    infcx.tcx,
                    cause.clone(),
                    goal.param_env,
                    goal.predicate,
                ));
            }
        };
        let normalized = infcx.deeply_resolve_ignoring_regions(infer_term);
        let normalization_was_ambiguous = match result.certainty {
            Certainty::Yes => NormalizationWasAmbiguous::No,
            Certainty::Maybe { .. } => {
                stalled_goals.push(result.goal);
                NormalizationWasAmbiguous::Yes
            }
        };
        Ok((normalized, normalization_was_ambiguous))
    });
    let value = value.try_fold_with(&mut folder)?;
    let obligations = stalled_goals
        .into_iter()
        .map(|goal| Obligation::new(infcx.tcx, cause.clone(), goal.param_env, goal.predicate))
        .collect();
    Ok(Normalized { value, obligations })
}

struct ReplaceAliasWithInfer<'me, 'tcx> {
    infcx: &'me InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &'me ObligationCause<'tcx>,
    obligations: PredicateObligations<'tcx>,
    universes: Vec<Option<UniverseIndex>>,
}

impl<'me, 'tcx> ReplaceAliasWithInfer<'me, 'tcx> {
    fn term_to_infer(&mut self, alias_term: ty::AliasTerm<'tcx>) -> ty::Term<'tcx> {
        let infcx = self.infcx;
        let infer_term = infcx.next_term_var_of_alias_kind(alias_term, self.cause.span);
        let obligation = Obligation::new(
            infcx.tcx,
            self.cause.clone(),
            self.param_env,
            ty::ProjectionClause { projection_term: alias_term, term: infer_term },
        );
        self.obligations.push(obligation);
        infer_term
    }
}

impl<'me, 'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceAliasWithInfer<'me, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: Binder<'tcx, T>,
    ) -> Binder<'tcx, T> {
        self.universes.push(None);
        let t = t.super_fold_with(self);
        self.universes.pop();
        t
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !self.cx().renormalize_rigid_aliases() && !ty.has_non_rigid_aliases() {
            return ty;
        }

        let ty = ty.super_fold_with(self);
        let ty::Alias(orig_is_rigid, alias) = *ty.kind() else { return ty };
        if !self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            return ty;
        }

        if ty.has_escaping_bound_vars() {
            let (replaced, ..) =
                BoundVarReplacer::replace_bound_vars(self.infcx, &mut self.universes, alias);
            // Keep the higher-ranked alias in the folded value; the fresh term is only
            // used to register its projection obligation.
            let _ = self.term_to_infer(replaced.into());
            ty
        } else {
            self.term_to_infer(alias.into()).expect_type()
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !self.cx().renormalize_rigid_aliases() && !ct.has_non_rigid_aliases() {
            return ct;
        }

        let ct = ct.super_fold_with(self);
        let ty::ConstKind::Alias(orig_is_rigid, alias_const) = ct.kind() else { return ct };
        if !self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            return ct;
        }

        if ct.has_escaping_bound_vars() {
            let (replaced, ..) =
                BoundVarReplacer::replace_bound_vars(self.infcx, &mut self.universes, alias_const);
            // Keep the higher-ranked alias in the folded value; the fresh term is only
            // used to register its projection obligation.
            let _ = self.term_to_infer(replaced.into());
            ct
        } else {
            self.term_to_infer(alias_const.into()).expect_const()
        }
    }
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
pub fn deeply_normalize<'tcx, T, E>(
    infcx: &InferCtxt<'tcx>,
    value: Unnormalized<'tcx, T>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &ObligationCause<'tcx>,
) -> Result<T, ThinVec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    assert!(!value.as_ref().skip_normalization().has_escaping_bound_vars());
    deeply_normalize_with_skipped_universes(infcx, value, vec![], param_env, cause)
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
///
/// Additionally takes a list of universes which represents the binders which have been
/// entered before passing `value` to the function. This is currently needed for
/// `normalize_erasing_regions`, which skips binders as it walks through a type.
pub fn deeply_normalize_with_skipped_universes<'tcx, T, E>(
    infcx: &InferCtxt<'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &ObligationCause<'tcx>,
) -> Result<T, ThinVec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    let (value, coroutine_goals) =
        deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
            infcx, value, universes, param_env, cause,
        )?;
    assert_eq!(coroutine_goals, vec![]);

    Ok(value)
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
///
/// Additionally takes a list of universes which represents the binders which have been
/// entered before passing `value` to the function. This is currently needed for
/// `normalize_erasing_regions`, which skips binders as it walks through a type.
///
/// This returns a set of stalled obligations involving coroutines if the typing mode of
/// the underlying infcx has any stalled coroutine def ids.
pub fn deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals<'tcx, T, E>(
    infcx: &InferCtxt<'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &ObligationCause<'tcx>,
) -> Result<(T, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), ThinVec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    let Normalized { value, obligations } = normalize_with_universes(
        infcx, value, universes, param_env, cause,
    )
    .map_err(|obligation| {
        thin_vec![E::from_solver_error(infcx, NextSolverError::TrueError(obligation))]
    })?;

    let mut fulfill_cx = FulfillmentCtxt::new(infcx);
    for pred in obligations {
        fulfill_cx.register_predicate_obligation(infcx, pred);
    }

    let errors = fulfill_cx.try_evaluate_obligations(infcx);
    if let TraitErrors::HasErrors(errors) = errors {
        return Err(errors);
    }

    let stalled_coroutine_goals = fulfill_cx
        .drain_stalled_obligations_for_coroutines(infcx)
        .into_iter()
        .map(|obl| obl.as_goal())
        .collect();

    let errors = fulfill_cx.collect_remaining_errors(infcx);
    if let TraitErrors::HasErrors(errors) = errors {
        return Err(errors);
    }

    Ok((value, stalled_coroutine_goals))
}

// Deeply normalize a value and return it
pub(crate) fn deeply_normalize_for_diagnostics<'tcx, T: TypeFoldable<TyCtxt<'tcx>>>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    t: T,
) -> T {
    t.fold_with(&mut DeeplyNormalizeForDiagnosticsFolder {
        infcx,
        cause: &ObligationCause::dummy(),
        param_env,
    })
}

/// A type folder struct.
///
/// This is isomorphic to what was previously called `At`. This should remain
/// specific to its use as a TypeFolder, and not expanded back into another
/// "God Object."
struct DeeplyNormalizeForDiagnosticsFolder<'a, 'tcx> {
    pub infcx: &'a InferCtxt<'tcx>,
    pub cause: &'a ObligationCause<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for DeeplyNormalizeForDiagnosticsFolder<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.infcx;
        let result: Result<_, ThinVec<ScrubbedTraitError<'tcx>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.infcx,
                Unnormalized::new_wip(ty),
                vec![None; ty.outer_exclusive_binder().as_usize()],
                self.param_env,
                self.cause,
            )
        });
        match result {
            Ok((ty, _)) => ty,
            Err(_) => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let infcx = self.infcx;
        let result: Result<_, ThinVec<ScrubbedTraitError<'tcx>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.infcx,
                Unnormalized::new_wip(ct),
                vec![None; ct.outer_exclusive_binder().as_usize()],
                self.param_env,
                self.cause,
            )
        });
        match result {
            Ok((ct, _)) => ct,
            Err(_) => ct.super_fold_with(self),
        }
    }
}
