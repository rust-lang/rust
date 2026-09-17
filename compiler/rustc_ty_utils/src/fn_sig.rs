use rustc_hir::def::DefKind;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypingMode, Unnormalized};
use rustc_span::def_id::{CRATE_DEF_ID, DefId};
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::bound_regions::{
    output_dependency_param_env, unconstrained_output_regions_after_normalization,
};
use rustc_trait_selection::traits::outlives_bounds::elaborate_projection_outlives;
use rustc_trait_selection::traits::{Obligation, ObligationCause, ObligationCtxt};

/// A function item may implement `Fn(P<'a>) -> P<'a>` for every `'a` even
/// when its declaration binds that lifetime early. This is only possible if
/// the output is determined by the complete input types and the lifetime
/// satisfies the declaration's requirements for every instantiation.
///
/// Keep this separate from `fn_sig`: explicit lifetime arguments still select
/// a particular instantiation for direct calls. Function items store no values
/// whose validity could depend on re-instantiating such a lifetime.
fn fn_sig_for_fn_traits<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> ty::EarlyBinder<'tcx, ty::PolyFnSig<'tcx>> {
    let declared = tcx.fn_sig(def_id);
    if !tcx.next_trait_solver_globally()
        || !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
    {
        return declared;
    }

    let generics = tcx.generics_of(def_id);
    let mut candidates: Vec<_> = generics
        .own_params
        .iter()
        .filter(|param| matches!(param.kind, ty::GenericParamDefKind::Lifetime))
        .collect();
    if candidates.is_empty() {
        return declared;
    }

    let clauses = tcx.clauses_of(def_id).instantiate_identity(tcx);
    candidates
        .retain(|param| requirements_hold_for_any_lifetime(tcx, def_id, param, &clauses.clauses));
    if candidates.is_empty() {
        return declared;
    }

    let signature = declared.skip_binder();
    let bind = |candidates: &[&ty::GenericParamDef]| {
        let mut bound_vars = signature.bound_vars().to_vec();
        let mut replacements = vec![None; generics.count()];
        for param in candidates {
            let kind = ty::BoundRegionKind::Named(param.def_id);
            let var = ty::BoundVar::from_usize(bound_vars.len());
            bound_vars.push(ty::BoundVariableKind::Region(kind));
            replacements[param.index as usize] = Some(ty::BoundRegion { var, kind });
        }
        let value = ty::fold_regions(tcx, signature.skip_binder(), |region, debruijn| {
            if let ty::ReEarlyParam(param) = region.kind()
                && let Some(bound) = replacements[param.index as usize]
            {
                ty::Region::new_bound(tcx, debruijn, bound)
            } else {
                region
            }
        });
        ty::Binder::bind_with_vars(value, tcx.mk_bound_variable_kinds(&bound_vars))
    };

    let tentative = bind(&candidates);
    let mut remaining = tcx.collect_output_late_bound_regions(
        tentative.map_bound(|sig| sig.inputs().to_vec()),
        tentative.output(),
    );
    if !remaining.is_empty() {
        let dependency = tentative.map_bound(|signature| {
            (
                tcx.mk_args_from_iter(
                    signature.inputs().iter().map(|&ty| ty::GenericArg::from(ty)),
                ),
                signature.output().into(),
            )
        });
        let cause =
            ObligationCause::misc(tcx.def_span(def_id), def_id.as_local().unwrap_or(CRATE_DEF_ID));
        remaining = unconstrained_output_regions_after_normalization(
            tcx,
            &cause,
            output_dependency_param_env(tcx, def_id),
            dependency,
        );
    }
    candidates.retain(|param| !remaining.contains(&ty::BoundRegionKind::Named(param.def_id)));
    if candidates.is_empty() { declared } else { ty::EarlyBinder::bind(tcx, bind(&candidates)) }
}

fn requirements_hold_for_any_lifetime<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    param: &ty::GenericParamDef,
    clauses: &[Unnormalized<'tcx, ty::Clause<'tcx>>],
) -> bool {
    let (dependent, independent): (Vec<_>, Vec<_>) =
        clauses.iter().map(|clause| clause.skip_norm_wip()).partition(|clause| {
            let mut depends = false;
            tcx.for_each_free_region(clause, |region| {
                depends |= matches!(region.kind(), ty::ReEarlyParam(p) if p.index == param.index);
            });
            depends
        });
    if dependent.is_empty() {
        return true;
    }

    // A requirement must not prove its own generalization. For example,
    // `View<'a>: Clone` is reusable only if the remaining environment or the
    // associated type's definition guarantees it for every lifetime.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new(&infcx);
    let cause =
        ObligationCause::misc(tcx.def_span(def_id), def_id.as_local().unwrap_or(CRATE_DEF_ID));
    let param_env = ty::ParamEnv::new(tcx, independent);
    let Ok(independent) = ocx.deeply_normalize(
        &cause,
        param_env,
        Unnormalized::new_wip(param_env.caller_bounds().collect::<Vec<_>>()),
    ) else {
        return false;
    };
    let param_env = elaborate_projection_outlives(tcx, &cause, ty::ParamEnv::new(tcx, independent));
    let signature = tcx.liberate_late_bound_regions(def_id, tcx.fn_sig(def_id).skip_binder());
    let Ok(inputs) =
        ocx.deeply_normalize(&cause, param_env, Unnormalized::new_wip(signature.inputs().to_vec()))
    else {
        return false;
    };
    // The early parameter is already universally quantified in this context.
    // The signature's inputs may supply its implied outlives requirements.
    ocx.register_obligations(
        dependent.into_iter().map(|clause| Obligation::new(tcx, cause.clone(), param_env, clause)),
    );
    ocx.evaluate_obligations_error_on_ambiguity().no_errors()
        && infcx.resolve_regions(cause.body_def_id, param_env, inputs).is_empty()
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.fn_sig_for_fn_traits = fn_sig_for_fn_traits;
}
