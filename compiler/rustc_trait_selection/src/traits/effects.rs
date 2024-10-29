use rustc_hir as hir;
use rustc_infer::infer::{BoundRegionConversionTime, DefineOpaqueTypes, InferCtxt};
use rustc_infer::traits::{ImplSource, Obligation, PredicateObligation};
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_middle::{span_bug, ty};
use rustc_type_ir::solve::NoSolution;
use thin_vec::ThinVec;

use super::SelectionContext;

pub type HostEffectObligation<'tcx> = Obligation<'tcx, ty::HostEffectPredicate<'tcx>>;

pub enum EvaluationFailure {
    Ambiguous,
    NoSolution,
}

pub fn evaluate_host_effect_obligation<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    if selcx.infcx.intercrate {
        span_bug!(
            obligation.cause.span,
            "should not select host obligation in old solver in intercrate mode"
        );
    }

    match evaluate_host_effect_from_bounds(selcx, obligation) {
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
    infcx: &InferCtxt<'tcx>,
    obligation: &HostEffectObligation<'tcx>,
    candidate: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, NoSolution> {
    if !candidate.skip_binder().host.satisfies(obligation.predicate.host) {
        return Err(NoSolution);
    }

    let candidate = infcx.instantiate_binder_with_fresh_vars(
        obligation.cause.span,
        BoundRegionConversionTime::HigherRankedType,
        candidate,
    );

    let mut nested = infcx
        .at(&obligation.cause, obligation.param_env)
        .eq(DefineOpaqueTypes::Yes, obligation.predicate.trait_ref, candidate.trait_ref)?
        .into_obligations();

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

    for predicate in obligation.param_env.caller_bounds() {
        let bound_predicate = predicate.kind();
        if let ty::ClauseKind::HostEffect(data) = predicate.kind().skip_binder() {
            let data = bound_predicate.rebind(data);
            if data.skip_binder().trait_ref.def_id != obligation.predicate.trait_ref.def_id {
                continue;
            }

            if !drcx.args_may_unify(
                obligation.predicate.trait_ref.args,
                data.skip_binder().trait_ref.args,
            ) {
                continue;
            }

            let is_match = infcx.probe(|_| match_candidate(infcx, obligation, data).is_ok());

            if is_match {
                if candidate.is_some() {
                    return Err(EvaluationFailure::Ambiguous);
                } else {
                    candidate = Some(data);
                }
            }
        }
    }

    if let Some(data) = candidate {
        Ok(match_candidate(infcx, obligation, data)
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
                    if tcx.constness(impl_.impl_def_id) != hir::Constness::Const {
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
                                    trait_ref.to_host_effect_clause(tcx, obligation.predicate.host),
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
