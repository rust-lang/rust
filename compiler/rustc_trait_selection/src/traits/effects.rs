use rustc_hir as hir;
use rustc_infer::infer::{BoundRegionConversionTime, DefineOpaqueTypes};
use rustc_infer::traits::{ImplSource, Obligation, PredicateObligation};
use rustc_middle::span_bug;
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_middle::ty::{self, TypingMode};
use rustc_type_ir::elaborate::elaborate;
use rustc_type_ir::solve::NoSolution;
use thin_vec::{ThinVec, thin_vec};

use super::SelectionContext;
use super::normalize::normalize_with_depth_to;

pub type HostEffectObligation<'tcx> = Obligation<'tcx, ty::HostEffectPredicate<'tcx>>;

pub enum EvaluationFailure {
    Ambiguous,
    NoSolution,
}

pub fn evaluate_host_effect_obligation<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    if matches!(selcx.infcx.typing_mode(), TypingMode::Coherence) {
        span_bug!(
            obligation.cause.span,
            "should not select host obligation in old solver in intercrate mode"
        );
    }

    // Force ambiguity for infer self ty.
    if obligation.predicate.self_ty().is_ty_var() {
        return Err(EvaluationFailure::Ambiguous);
    }

    match evaluate_host_effect_from_bounds(selcx, obligation) {
        Ok(result) => return Ok(result),
        Err(EvaluationFailure::Ambiguous) => return Err(EvaluationFailure::Ambiguous),
        Err(EvaluationFailure::NoSolution) => {}
    }

    match evaluate_host_effect_from_item_bounds(selcx, obligation) {
        Ok(result) => return Ok(result),
        Err(EvaluationFailure::Ambiguous) => return Err(EvaluationFailure::Ambiguous),
        Err(EvaluationFailure::NoSolution) => {}
    }

    match evaluate_host_effect_from_selection_candiate(selcx, obligation) {
        Ok(result) => return Ok(result),
        Err(EvaluationFailure::Ambiguous) => return Err(EvaluationFailure::Ambiguous),
        Err(EvaluationFailure::NoSolution) => {}
    }

    Err(EvaluationFailure::NoSolution)
}

fn match_candidate<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
    candidate: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
    candidate_is_unnormalized: bool,
    more_nested: impl FnOnce(&mut SelectionContext<'_, 'tcx>, &mut ThinVec<PredicateObligation<'tcx>>),
) -> Result<ThinVec<PredicateObligation<'tcx>>, NoSolution> {
    if !candidate.skip_binder().constness.satisfies(obligation.predicate.constness) {
        return Err(NoSolution);
    }

    let mut candidate = selcx.infcx.instantiate_binder_with_fresh_vars(
        obligation.cause.span,
        BoundRegionConversionTime::HigherRankedType,
        candidate,
    );

    let mut nested = thin_vec![];

    // Unlike param-env bounds, item bounds may not be normalized.
    if candidate_is_unnormalized {
        candidate = normalize_with_depth_to(
            selcx,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth,
            candidate,
            &mut nested,
        );
    }

    nested.extend(
        selcx
            .infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(DefineOpaqueTypes::Yes, obligation.predicate.trait_ref, candidate.trait_ref)?
            .into_obligations(),
    );

    more_nested(selcx, &mut nested);

    for nested in &mut nested {
        nested.set_depth_from_parent(obligation.recursion_depth);
    }

    Ok(nested)
}

fn evaluate_host_effect_from_bounds<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    let infcx = selcx.infcx;
    let drcx = DeepRejectCtxt::relate_rigid_rigid(selcx.tcx());
    let mut candidate = None;

    for clause in obligation.param_env.caller_bounds() {
        let bound_clause = clause.kind();
        let ty::ClauseKind::HostEffect(data) = bound_clause.skip_binder() else {
            continue;
        };
        let data = bound_clause.rebind(data);
        if data.skip_binder().trait_ref.def_id != obligation.predicate.trait_ref.def_id {
            continue;
        }

        if !drcx
            .args_may_unify(obligation.predicate.trait_ref.args, data.skip_binder().trait_ref.args)
        {
            continue;
        }

        let is_match =
            infcx.probe(|_| match_candidate(selcx, obligation, data, false, |_, _| {}).is_ok());

        if is_match {
            if candidate.is_some() {
                return Err(EvaluationFailure::Ambiguous);
            } else {
                candidate = Some(data);
            }
        }
    }

    if let Some(data) = candidate {
        Ok(match_candidate(selcx, obligation, data, false, |_, _| {})
            .expect("candidate matched before, so it should match again"))
    } else {
        Err(EvaluationFailure::NoSolution)
    }
}

fn evaluate_host_effect_from_item_bounds<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    let infcx = selcx.infcx;
    let tcx = infcx.tcx;
    let drcx = DeepRejectCtxt::relate_rigid_rigid(selcx.tcx());
    let mut candidate = None;

    let mut consider_ty = obligation.predicate.self_ty();
    while let ty::Alias(kind @ (ty::Projection | ty::Opaque), alias_ty) = *consider_ty.kind() {
        if tcx.is_conditionally_const(alias_ty.def_id) {
            for clause in elaborate(
                tcx,
                tcx.explicit_implied_const_bounds(alias_ty.def_id)
                    .iter_instantiated_copied(tcx, alias_ty.args)
                    .map(|(trait_ref, _)| {
                        trait_ref.to_host_effect_clause(tcx, obligation.predicate.constness)
                    }),
            ) {
                let bound_clause = clause.kind();
                let ty::ClauseKind::HostEffect(data) = bound_clause.skip_binder() else {
                    unreachable!("should not elaborate non-HostEffect from HostEffect")
                };
                let data = bound_clause.rebind(data);
                if data.skip_binder().trait_ref.def_id != obligation.predicate.trait_ref.def_id {
                    continue;
                }

                if !drcx.args_may_unify(
                    obligation.predicate.trait_ref.args,
                    data.skip_binder().trait_ref.args,
                ) {
                    continue;
                }

                let is_match = infcx
                    .probe(|_| match_candidate(selcx, obligation, data, true, |_, _| {}).is_ok());

                if is_match {
                    if candidate.is_some() {
                        return Err(EvaluationFailure::Ambiguous);
                    } else {
                        candidate = Some((data, alias_ty));
                    }
                }
            }
        }

        if kind != ty::Projection {
            break;
        }

        consider_ty = alias_ty.self_ty();
    }

    if let Some((data, alias_ty)) = candidate {
        Ok(match_candidate(selcx, obligation, data, true, |selcx, nested| {
            // An alias bound only holds if we also check the const conditions
            // of the alias, so we need to register those, too.
            let const_conditions = normalize_with_depth_to(
                selcx,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth,
                tcx.const_conditions(alias_ty.def_id).instantiate(tcx, alias_ty.args),
                nested,
            );
            nested.extend(const_conditions.into_iter().map(|(trait_ref, _)| {
                obligation
                    .with(tcx, trait_ref.to_host_effect_clause(tcx, obligation.predicate.constness))
            }));
        })
        .expect("candidate matched before, so it should match again"))
    } else {
        Err(EvaluationFailure::NoSolution)
    }
}

fn evaluate_host_effect_from_selection_candiate<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    let tcx = selcx.tcx();
    selcx.infcx.commit_if_ok(|_| {
        match selcx.select(&obligation.with(tcx, obligation.predicate.trait_ref)) {
            Ok(None) => Err(EvaluationFailure::Ambiguous),
            Err(_) => Err(EvaluationFailure::NoSolution),
            Ok(Some(source)) => match source {
                ImplSource::UserDefined(impl_) => {
                    if tcx.impl_trait_header(impl_.impl_def_id).unwrap().constness
                        != hir::Constness::Const
                    {
                        return Err(EvaluationFailure::NoSolution);
                    }

                    let mut nested = impl_.nested;
                    nested.extend(
                        tcx.const_conditions(impl_.impl_def_id)
                            .instantiate(tcx, impl_.args)
                            .into_iter()
                            .map(|(trait_ref, _)| {
                                obligation.with(
                                    tcx,
                                    trait_ref
                                        .to_host_effect_clause(tcx, obligation.predicate.constness),
                                )
                            }),
                    );

                    for nested in &mut nested {
                        nested.set_depth_from_parent(obligation.recursion_depth);
                    }

                    Ok(nested)
                }
                _ => Err(EvaluationFailure::NoSolution),
            },
        }
    })
}
