use rustc_infer::infer::InferCtxt;
use rustc_infer::infer::at::At;
use rustc_infer::traits::solve::Goal;
use rustc_infer::traits::{
    FromSolverError, Normalized, Obligation, PredicateObligations, TraitEngine,
};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, AliasTerm, Binder, Flags, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt, UniverseIndex, Unnormalized,
};
use rustc_next_trait_solver::normalize::{NormalizationFolder, NormalizationWasAmbiguous};
use rustc_next_trait_solver::solve::{NoSolution, SolverDelegateEvalExt};

use super::{FulfillmentCtxt, NextSolverError};
use crate::solve::{Certainty, SolverDelegate};
use crate::traits::{BoundVarReplacer, ScrubbedTraitError};

/// see `normalize_with_universes`.
pub fn normalize<'tcx, T>(at: At<'_, 'tcx>, value: Unnormalized<'tcx, T>) -> Normalized<'tcx, T>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    normalize_with_universes(at, value, vec![])
}

/// Like `deeply_normalize`, but we handle ambiguity and inference variables in this routine.
/// The behavior should be same as the old solver.
/// For error, we return an infer var plus the failed obligation.
/// For ambiguity, we have two cases:
///   - has_escaping_bound_vars: return the original alias.
///   - otherwise: return the normalized result. It can be (partially) inferred
///     even if the evaluation result is ambiguous.
fn normalize_with_universes<'tcx, T>(
    at: At<'_, 'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
) -> Normalized<'tcx, T>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    let infcx = at.infcx;
    let value = value.skip_normalization();
    let value = infcx.resolve_vars_if_possible(value);
    let original_value = value.clone();
    let mut stalled_goals = vec![];
    let mut folder = NormalizationFolder::new(infcx, universes.clone(), |alias_term| {
        let delegate = <&SolverDelegate<'tcx>>::from(infcx);
        let infer_term = delegate.next_term_var_for_alias(alias_term, at.cause.span);
        let predicate = ty::PredicateKind::AliasRelate(
            alias_term.to_term(infcx.tcx),
            infer_term.into(),
            ty::AliasRelationDirection::Equate,
        );
        let goal = Goal::new(infcx.tcx, at.param_env, predicate);
        let result = match delegate.evaluate_root_goal(goal, at.cause.span, None) {
            Ok(result) => result,
            Err(err) => return Err(err),
        };
        let normalized = infcx.resolve_vars_if_possible(infer_term);
        let normalization_was_ambiguous = match result.certainty {
            Certainty::Yes => NormalizationWasAmbiguous::No,
            Certainty::Maybe { .. } => {
                stalled_goals.push(infcx.resolve_vars_if_possible(goal));
                NormalizationWasAmbiguous::Yes
            }
        };
        Ok((normalized, normalization_was_ambiguous))
    });

    match value.try_fold_with(&mut folder) {
        Ok(value) => {
            let obligations = stalled_goals
                .into_iter()
                .map(|goal| {
                    Obligation::new(infcx.tcx, at.cause.clone(), goal.param_env, goal.predicate)
                })
                .collect();
            Normalized { value, obligations }
        }
        Err(NoSolution) => {
            let mut replacer =
                ReplaceAliasWithInfer { at, obligations: Default::default(), universes };
            let value = original_value.fold_with(&mut replacer);
            Normalized { value, obligations: replacer.obligations }
        }
    }
}

struct ReplaceAliasWithInfer<'me, 'tcx> {
    at: At<'me, 'tcx>,
    obligations: PredicateObligations<'tcx>,
    universes: Vec<Option<UniverseIndex>>,
}

impl<'me, 'tcx> ReplaceAliasWithInfer<'me, 'tcx> {
    fn alias_term_to_infer(&mut self, alias_term: ty::AliasTerm<'tcx>) -> ty::Term<'tcx> {
        let infcx = self.at.infcx;
        let infer_term = infcx.next_term_var_for_alias(alias_term, self.at.cause.span);
        let obligation = Obligation::new(
            infcx.tcx,
            self.at.cause.clone(),
            self.at.param_env,
            ty::PredicateKind::AliasRelate(
                alias_term.to_term(infcx.tcx),
                infer_term.into(),
                ty::AliasRelationDirection::Equate,
            ),
        );
        self.obligations.push(obligation);
        infer_term
    }
}

impl<'me, 'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceAliasWithInfer<'me, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.at.infcx.tcx
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
        if !ty.has_aliases() {
            return ty;
        }

        let ty = ty.super_fold_with(self);
        let ty::Alias(alias_ty) = *ty.kind() else { return ty };
        if alias_ty.has_escaping_bound_vars() {
            let (replaced, ..) =
                BoundVarReplacer::replace_bound_vars(self.at.infcx, &mut self.universes, alias_ty);
            let _ = self.alias_term_to_infer(replaced.into());
            ty
        } else {
            self.alias_term_to_infer(alias_ty.into()).expect_type()
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !ct.has_aliases() {
            return ct;
        }

        let ct = ct.super_fold_with(self);
        let ty::ConstKind::Unevaluated(uv) = ct.kind() else { return ct };

        if ct.has_escaping_bound_vars() {
            let (replaced, ..) =
                BoundVarReplacer::replace_bound_vars(self.at.infcx, &mut self.universes, uv);
            let _ =
                self.alias_term_to_infer(AliasTerm::from_unevaluated_const(self.cx(), replaced));
            ct
        } else {
            self.alias_term_to_infer(AliasTerm::from_unevaluated_const(self.cx(), uv))
                .expect_const()
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.allow_normalization() { p.super_fold_with(self) } else { p }
    }
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
pub fn deeply_normalize<'tcx, T, E>(
    at: At<'_, 'tcx>,
    value: Unnormalized<'tcx, T>,
) -> Result<T, Vec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    assert!(!value.as_ref().skip_normalization().has_escaping_bound_vars());
    deeply_normalize_with_skipped_universes(at, value, vec![])
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
///
/// Additionally takes a list of universes which represents the binders which have been
/// entered before passing `value` to the function. This is currently needed for
/// `normalize_erasing_regions`, which skips binders as it walks through a type.
pub fn deeply_normalize_with_skipped_universes<'tcx, T, E>(
    at: At<'_, 'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
) -> Result<T, Vec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    let (value, coroutine_goals) =
        deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
            at, value, universes,
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
    at: At<'_, 'tcx>,
    value: Unnormalized<'tcx, T>,
    universes: Vec<Option<UniverseIndex>>,
) -> Result<(T, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), Vec<E>>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>>,
{
    let Normalized { value, obligations } = normalize_with_universes(at, value, universes);

    let mut fulfill_cx = FulfillmentCtxt::new(at.infcx);
    for pred in obligations {
        fulfill_cx.register_predicate_obligation(at.infcx, pred);
    }

    let errors = fulfill_cx.try_evaluate_obligations(at.infcx);
    if !errors.is_empty() {
        return Err(errors);
    }

    let stalled_coroutine_goals = fulfill_cx
        .drain_stalled_obligations_for_coroutines(at.infcx)
        .into_iter()
        .map(|obl| obl.as_goal())
        .collect();

    let errors = fulfill_cx.collect_remaining_errors(at.infcx);
    if !errors.is_empty() {
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
        at: infcx.at(&ObligationCause::dummy(), param_env),
    })
}

struct DeeplyNormalizeForDiagnosticsFolder<'a, 'tcx> {
    at: At<'a, 'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for DeeplyNormalizeForDiagnosticsFolder<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.at.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.at.infcx;
        let result: Result<_, Vec<ScrubbedTraitError<'tcx>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.at,
                Unnormalized::new_wip(ty),
                vec![None; ty.outer_exclusive_binder().as_usize()],
            )
        });
        match result {
            Ok((ty, _)) => ty,
            Err(_) => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let infcx = self.at.infcx;
        let result: Result<_, Vec<ScrubbedTraitError<'tcx>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.at,
                Unnormalized::new_wip(ct),
                vec![None; ct.outer_exclusive_binder().as_usize()],
            )
        });
        match result {
            Ok((ct, _)) => ct,
            Err(_) => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.allow_normalization() { p.super_fold_with(self) } else { p }
    }
}
