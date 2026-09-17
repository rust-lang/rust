use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::infer::canonical::QueryRegionConstraint;
use rustc_infer::infer::{InferOk, TyCtxtInferExt};
use rustc_infer::traits::query::type_op::ImpliedOutlivesBounds;
use rustc_macros::extension;
use rustc_middle::infer::canonical::{OriginalQueryValues, QueryRegionConstraints};
pub use rustc_middle::traits::query::OutlivesBound;
use rustc_middle::ty::{
    self, ParamEnv, Ty, TyCtxt, TypeVisitableExt, TypingMode, Unnormalized, Upcast,
};
use rustc_span::def_id::LocalDefId;
use tracing::instrument;

use crate::infer::InferCtxt;
use crate::regions::InferCtxtRegionExt;
use crate::traits::{ObligationCause, ObligationCtxt};

/// Preserve the declaration bounds of a projection after an environment
/// equality replaces it by another type. Quantified equalities may only be
/// used after their well-formedness premises hold for every instantiation.
pub fn elaborate_projection_outlives<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: &ObligationCause<'tcx>,
    mut param_env: ParamEnv<'tcx>,
) -> ParamEnv<'tcx> {
    if !tcx.next_trait_solver_globally() || param_env.has_infer() {
        return param_env;
    }
    let mut pending = Vec::new();
    for projection in param_env.caller_bounds().filter_map(|clause| clause.as_projection_clause()) {
        let value = projection.skip_binder();
        let Some(output) = value.term.as_type() else { continue };
        let alias = value.projection_term.expect_ty();
        let ty::AliasTyKind::Projection { def_id } = alias.kind else { continue };
        for &(declaration, _) in tcx.explicit_item_self_bounds(def_id).skip_binder() {
            let declaration = declaration.kind();
            let shifted = tcx
                .shift_bound_var_indices(projection.bound_vars().len(), declaration.skip_binder());
            let value =
                ty::EarlyBinder::bind(tcx, shifted).instantiate(tcx, alias.args).skip_norm_wip();
            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                projection.bound_vars().iter().chain(declaration.bound_vars()),
            );
            let declaration: ty::Clause<'tcx> =
                ty::Binder::bind_with_vars(value, bound_vars).upcast(tcx);
            let declarations: Vec<_> =
                super::elaborate(tcx, [declaration]).filter_only_self().collect();
            let bounds: Vec<ty::Clause<'tcx>> = declarations
                .iter()
                .filter_map(|clause| clause.as_type_outlives_clause())
                .map(|bound| {
                    bound
                        .map_bound(|ty::OutlivesClause(_, region)| {
                            ty::ClauseKind::TypeOutlives(ty::OutlivesClause(output, region))
                        })
                        .upcast(tcx)
                })
                .collect();
            if !bounds.is_empty() {
                let premises: Vec<_> = declarations
                    .into_iter()
                    .map(|declaration| {
                        declaration.kind().map_bound(|clause| (ty::AliasTerm::from(alias), clause))
                    })
                    .collect();
                pending.push((premises, bounds));
            }
        }
    }
    if pending.is_empty() {
        return param_env;
    }
    let mut clauses: FxIndexSet<_> = param_env.caller_bounds().collect();
    loop {
        let before = pending.len();
        pending.retain(|(premises, bounds)| {
            let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            let ocx = ObligationCtxt::new(&infcx);
            for premise in premises.iter().filter(|premise| premise.has_bound_vars()) {
                let proven = infcx.enter_forall(*premise, |(alias, declaration)| {
                    infcx.insert_placeholder_assumptions(
                        infcx.universe(),
                        Some(ty::region_constraint::Assumptions::empty()),
                    );
                    let Some(obligations) = super::wf::obligations(
                        &infcx,
                        param_env,
                        cause.body_def_id,
                        0,
                        alias.to_term(tcx, ty::IsRigid::No),
                        cause.span,
                    ) else {
                        return false;
                    };
                    ocx.register_obligations(obligations);
                    // Only the declaration's type arguments introduce scoped
                    // WF assumptions. Its supertrait bounds are conclusions,
                    // so requiring them here would make the proof circular.
                    let terms: Vec<_> = match declaration {
                        ty::ClauseKind::Trait(clause) => {
                            clause.trait_ref.args.iter().filter_map(|arg| arg.as_term()).collect()
                        }
                        ty::ClauseKind::TypeOutlives(ty::OutlivesClause(ty, _)) => vec![ty.into()],
                        _ => vec![],
                    };
                    for term in terms {
                        let Some(obligations) = super::wf::obligations(
                            &infcx,
                            param_env,
                            cause.body_def_id,
                            0,
                            term,
                            cause.span,
                        ) else {
                            return false;
                        };
                        ocx.register_obligations(obligations);
                    }
                    ocx.evaluate_obligations_error_on_ambiguity().no_errors()
                });
                if !proven {
                    return true;
                }
            }
            let Ok(bounds) =
                ocx.deeply_normalize(cause, param_env, Unnormalized::new_wip(bounds.clone()))
            else {
                return true;
            };
            if !infcx.resolve_regions(cause.body_def_id, param_env, []).is_empty() {
                return true;
            }
            let Ok(bounds) = infcx.deeply_resolve_via_region_graph(bounds) else {
                return true;
            };
            clauses.extend(super::elaborate(tcx, bounds));
            false
        });
        param_env = ParamEnv::new(tcx, clauses.iter().copied());
        if pending.len() == before || pending.is_empty() {
            return param_env;
        }
    }
}

/// Implied bounds are region relationships that we deduce
/// automatically. The idea is that (e.g.) a caller must check that a
/// function's argument types are well-formed immediately before
/// calling that fn, and hence the *callee* can assume that its
/// argument types are well-formed. This may imply certain relationships
/// between generic parameters. For example:
/// ```
/// fn foo<T>(x: &T) {}
/// ```
/// can only be called with a `'a` and `T` such that `&'a T` is WF.
/// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
///
/// # Parameters
///
/// - `param_env`, the where-clauses in scope
/// - `body_def_id`, the body_def_id to use when normalizing assoc types.
///   Note that this may cause outlives obligations to be injected
///   into the inference context with this body-id.
/// - `ty`, the type that we are supposed to assume is WF.
#[instrument(level = "debug", skip(infcx, param_env, body_def_id), ret)]
fn implied_outlives_bounds<'a, 'tcx>(
    infcx: &'a InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_def_id: LocalDefId,
    ty: Ty<'tcx>,
    disable_implied_bounds_hack: bool,
) -> Vec<OutlivesBound<'tcx>> {
    let ty = infcx.deeply_resolve_via_unification_table(ty);

    // We do not expect existential variables in implied bounds.
    // We may however encounter unconstrained lifetime variables
    // in very rare cases.
    //
    // See `ui/implied-bounds/implied-bounds-unconstrained-2.rs` for
    // an example.
    assert!(!ty.has_non_region_infer());

    let mut canonical_var_values = OriginalQueryValues::default();
    let input = ImpliedOutlivesBounds { ty };
    let canonical = infcx.canonicalize_query(param_env.and(input), &mut canonical_var_values);
    let implied_bounds_result =
        infcx.tcx.implied_outlives_bounds((canonical, disable_implied_bounds_hack));
    let Ok(canonical_result) = implied_bounds_result else {
        return vec![];
    };

    let mut constraints = QueryRegionConstraints::default();
    let span = infcx.tcx.def_span(body_def_id);
    let Ok(InferOk { value: mut bounds, obligations }) = infcx
        .instantiate_nll_query_response_and_region_obligations(
            &ObligationCause::dummy_with_span(span),
            param_env,
            &canonical_var_values,
            canonical_result,
            &mut constraints,
        )
    else {
        return vec![];
    };
    assert_eq!(obligations.len(), 0);

    // Because of #109628, we may have unexpected placeholders. Ignore them!
    // FIXME(#109628): panic in this case once the issue is fixed.
    bounds.retain(|bound| !bound.has_placeholders());

    if !constraints.is_empty() {
        // FIXME(higher_ranked_auto): Should we register assumptions here?
        // We otherwise would get spurious errors if normalizing an implied
        // outlives bound required proving some higher-ranked coroutine obl.
        let QueryRegionConstraints { constraints, assumptions: _, solver_region_constraints } =
            constraints;
        for constraint in solver_region_constraints {
            infcx.register_solver_region_constraint(constraint);
        }
        let cause = ObligationCause::misc(span, body_def_id);
        for &QueryRegionConstraint { constraint, visible_for_leak_check: vis, .. } in &constraints {
            match constraint {
                ty::RegionConstraint::Outlives(predicate) => {
                    infcx.register_outlives_constraint(predicate, vis, &cause)
                }
                ty::RegionConstraint::Eq(predicate) => {
                    infcx.register_region_eq_constraint(predicate, vis, &cause)
                }
            }
        }
    };

    bounds
}

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Do *NOT* call this directly. You probably want to construct a `OutlivesEnvironment`
    /// instead if you're interested in the implied bounds for a given signature.
    fn implied_bounds_tys<Tys: IntoIterator<Item = Ty<'tcx>>>(
        &self,
        body_def_id: LocalDefId,
        param_env: ParamEnv<'tcx>,
        tys: Tys,
        disable_implied_bounds_hack: bool,
    ) -> impl Iterator<Item = OutlivesBound<'tcx>> {
        tys.into_iter().flat_map(move |ty| {
            implied_outlives_bounds(self, param_env, body_def_id, ty, disable_implied_bounds_hack)
        })
    }
}
