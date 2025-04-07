use rustc_hir::{self as hir, LangItem};
use rustc_infer::infer::{BoundRegionConversionTime, DefineOpaqueTypes};
use rustc_infer::traits::{
    ImplDerivedHostCause, ImplSource, Obligation, ObligationCauseCode, PredicateObligation,
};
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

    let ref obligation = selcx.infcx.resolve_vars_if_possible(obligation.clone());

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

    match evaluate_host_effect_from_builtin_impls(selcx, obligation) {
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

fn evaluate_host_effect_from_builtin_impls<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    match selcx.tcx().as_lang_item(obligation.predicate.def_id()) {
        Some(LangItem::Destruct) => evaluate_host_effect_for_destruct_goal(selcx, obligation),
        _ => Err(EvaluationFailure::NoSolution),
    }
}

// NOTE: Keep this in sync with `const_conditions_for_destruct` in the new solver.
fn evaluate_host_effect_for_destruct_goal<'tcx>(
    selcx: &mut SelectionContext<'_, 'tcx>,
    obligation: &HostEffectObligation<'tcx>,
) -> Result<ThinVec<PredicateObligation<'tcx>>, EvaluationFailure> {
    let tcx = selcx.tcx();
    let destruct_def_id = tcx.require_lang_item(LangItem::Destruct, None);
    let self_ty = obligation.predicate.self_ty();

    let const_conditions = match *self_ty.kind() {
        // An ADT is `~const Destruct` only if all of the fields are,
        // *and* if there is a `Drop` impl, that `Drop` impl is also `~const`.
        ty::Adt(adt_def, args) => {
            let mut const_conditions: ThinVec<_> = adt_def
                .all_fields()
                .map(|field| ty::TraitRef::new(tcx, destruct_def_id, [field.ty(tcx, args)]))
                .collect();
            match adt_def.destructor(tcx).map(|dtor| tcx.constness(dtor.did)) {
                // `Drop` impl exists, but it's not const. Type cannot be `~const Destruct`.
                Some(hir::Constness::NotConst) => return Err(EvaluationFailure::NoSolution),
                // `Drop` impl exists, and it's const. Require `Ty: ~const Drop` to hold.
                Some(hir::Constness::Const) => {
                    let drop_def_id = tcx.require_lang_item(LangItem::Drop, None);
                    let drop_trait_ref = ty::TraitRef::new(tcx, drop_def_id, [self_ty]);
                    const_conditions.push(drop_trait_ref);
                }
                // No `Drop` impl, no need to require anything else.
                None => {}
            }
            const_conditions
        }

        ty::Array(ty, _) | ty::Pat(ty, _) | ty::Slice(ty) => {
            thin_vec![ty::TraitRef::new(tcx, destruct_def_id, [ty])]
        }

        ty::Tuple(tys) => {
            tys.iter().map(|field_ty| ty::TraitRef::new(tcx, destruct_def_id, [field_ty])).collect()
        }

        // Trivially implement `~const Destruct`
        ty::Bool
        | ty::Char
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::Str
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::FnDef(..)
        | ty::FnPtr(..)
        | ty::Never
        | ty::Infer(ty::InferTy::FloatVar(_) | ty::InferTy::IntVar(_))
        | ty::Error(_) => thin_vec![],

        // Coroutines and closures could implement `~const Drop`,
        // but they don't really need to right now.
        ty::Closure(_, _)
        | ty::CoroutineClosure(_, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(_, _) => return Err(EvaluationFailure::NoSolution),

        // FIXME(unsafe_binders): Unsafe binders could implement `~const Drop`
        // if their inner type implements it.
        ty::UnsafeBinder(_) => return Err(EvaluationFailure::NoSolution),

        ty::Dynamic(..) | ty::Param(_) | ty::Alias(..) | ty::Placeholder(_) | ty::Foreign(_) => {
            return Err(EvaluationFailure::NoSolution);
        }

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            panic!("unexpected type `{self_ty:?}`")
        }
    };

    Ok(const_conditions
        .into_iter()
        .map(|trait_ref| {
            obligation.with(
                tcx,
                ty::Binder::dummy(trait_ref)
                    .to_host_effect_clause(tcx, obligation.predicate.constness),
            )
        })
        .collect())
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
                            .map(|(trait_ref, span)| {
                                Obligation::new(
                                    tcx,
                                    obligation.cause.clone().derived_host_cause(
                                        ty::Binder::dummy(obligation.predicate),
                                        |derived| {
                                            ObligationCauseCode::ImplDerivedHost(Box::new(
                                                ImplDerivedHostCause {
                                                    derived,
                                                    impl_def_id: impl_.impl_def_id,
                                                    span,
                                                },
                                            ))
                                        },
                                    ),
                                    obligation.param_env,
                                    trait_ref
                                        .to_host_effect_clause(tcx, obligation.predicate.constness),
                                )
                            }),
                    );

                    Ok(nested)
                }
                _ => Err(EvaluationFailure::NoSolution),
            },
        }
    })
}
