use rustc_infer::infer::canonical::CanonicalQueryInput;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::traits::query::OutlivesBound;
use rustc_infer::traits::query::type_op::ImpliedOutlivesBounds;
use rustc_middle::infer::canonical::CanonicalQueryResponse;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, TypeFolder, TypeVisitableExt};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_type_ir::outlives::{Component, push_outlives_components};
use smallvec::{SmallVec, smallvec};
use tracing::debug;

use crate::traits::query::NoSolution;
use crate::traits::{ObligationCtxt, wf};

impl<'tcx> super::QueryTypeOp<'tcx> for ImpliedOutlivesBounds<'tcx> {
    type QueryResponse = Vec<OutlivesBound<'tcx>>;

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        // Don't go into the query for things that can't possibly have lifetimes.
        match key.value.ty.kind() {
            ty::Tuple(elems) if elems.is_empty() => Some(vec![]),
            ty::Never | ty::Str | ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                Some(vec![])
            }
            _ => None,
        }
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        if tcx.sess.opts.unstable_opts.no_implied_bounds_compat {
            tcx.implied_outlives_bounds(canonicalized)
        } else {
            tcx.implied_outlives_bounds_compat(canonicalized)
        }
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        if ocx.infcx.tcx.sess.opts.unstable_opts.no_implied_bounds_compat {
            compute_implied_outlives_bounds_inner(ocx, key.param_env, key.value.ty)
        } else {
            compute_implied_outlives_bounds_compat_inner(ocx, key.param_env, key.value.ty)
        }
    }
}

pub fn compute_implied_outlives_bounds_inner<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<OutlivesBound<'tcx>>, NoSolution> {
    let normalize_op = |ty| -> Result<_, NoSolution> {
        // We must normalize the type so we can compute the right outlives components.
        // for example, if we have some constrained param type like `T: Trait<Out = U>`,
        // and we know that `&'a T::Out` is WF, then we want to imply `U: 'a`.
        let ty = ocx
            .deeply_normalize(&ObligationCause::dummy(), param_env, ty)
            .map_err(|_| NoSolution)?;
        if !ocx.select_all_or_error().is_empty() {
            return Err(NoSolution);
        }
        let ty = OpportunisticRegionResolver::new(&ocx.infcx).fold_ty(ty);
        Ok(ty)
    };

    // Sometimes when we ask what it takes for T: WF, we get back that
    // U: WF is required; in that case, we push U onto this stack and
    // process it next. Because the resulting predicates aren't always
    // guaranteed to be a subset of the original type, so we need to store the
    // WF args we've computed in a set.
    let mut checked_wf_args = rustc_data_structures::fx::FxHashSet::default();
    let mut wf_args = vec![ty.into(), normalize_op(ty)?.into()];

    let mut outlives_bounds: Vec<OutlivesBound<'tcx>> = vec![];

    while let Some(arg) = wf_args.pop() {
        if !checked_wf_args.insert(arg) {
            continue;
        }

        // From the full set of obligations, just filter down to the region relationships.
        for obligation in
            wf::unnormalized_obligations(ocx.infcx, param_env, arg).into_iter().flatten()
        {
            assert!(!obligation.has_escaping_bound_vars());
            let Some(pred) = obligation.predicate.kind().no_bound_vars() else {
                continue;
            };
            match pred {
                // FIXME(const_generics): Make sure that `<'a, 'b, const N: &'a &'b u32>` is sound
                // if we ever support that
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                | ty::PredicateKind::DynCompatible(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::NormalizesTo(..)
                | ty::PredicateKind::AliasRelate(..) => {}

                // We need to search through *all* WellFormed predicates
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
                    wf_args.push(arg);
                }

                // We need to register region relationships
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(
                    ty::OutlivesPredicate(r_a, r_b),
                )) => outlives_bounds.push(OutlivesBound::RegionSubRegion(r_b, r_a)),

                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    ty_a,
                    r_b,
                ))) => {
                    let ty_a = normalize_op(ty_a)?;
                    let mut components = smallvec![];
                    push_outlives_components(ocx.infcx.tcx, ty_a, &mut components);
                    outlives_bounds.extend(implied_bounds_from_components(r_b, components))
                }
            }
        }
    }

    Ok(outlives_bounds)
}

pub fn compute_implied_outlives_bounds_compat_inner<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<OutlivesBound<'tcx>>, NoSolution> {
    let tcx = ocx.infcx.tcx;

    // Sometimes when we ask what it takes for T: WF, we get back that
    // U: WF is required; in that case, we push U onto this stack and
    // process it next. Because the resulting predicates aren't always
    // guaranteed to be a subset of the original type, so we need to store the
    // WF args we've computed in a set.
    let mut checked_wf_args = rustc_data_structures::fx::FxHashSet::default();
    let mut wf_args = vec![ty.into()];

    let mut outlives_bounds: Vec<ty::OutlivesPredicate<'tcx, ty::GenericArg<'tcx>>> = vec![];

    while let Some(arg) = wf_args.pop() {
        if !checked_wf_args.insert(arg) {
            continue;
        }

        // Compute the obligations for `arg` to be well-formed. If `arg` is
        // an unresolved inference variable, just instantiated an empty set
        // -- because the return type here is going to be things we *add*
        // to the environment, it's always ok for this set to be smaller
        // than the ultimate set. (Note: normally there won't be
        // unresolved inference variables here anyway, but there might be
        // during typeck under some circumstances.)
        //
        // FIXME(@lcnr): It's not really "always fine", having fewer implied
        // bounds can be backward incompatible, e.g. #101951 was caused by
        // us not dealing with inference vars in `TypeOutlives` predicates.
        let obligations = wf::obligations(ocx.infcx, param_env, CRATE_DEF_ID, 0, arg, DUMMY_SP)
            .unwrap_or_default();

        for obligation in obligations {
            debug!(?obligation);
            assert!(!obligation.has_escaping_bound_vars());

            // While these predicates should all be implied by other parts of
            // the program, they are still relevant as they may constrain
            // inference variables, which is necessary to add the correct
            // implied bounds in some cases, mostly when dealing with projections.
            //
            // Another important point here: we only register `Projection`
            // predicates, since otherwise we might register outlives
            // predicates containing inference variables, and we don't
            // learn anything new from those.
            if obligation.predicate.has_non_region_infer() {
                match obligation.predicate.kind().skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                    | ty::PredicateKind::AliasRelate(..) => {
                        ocx.register_obligation(obligation.clone());
                    }
                    _ => {}
                }
            }

            let pred = match obligation.predicate.kind().no_bound_vars() {
                None => continue,
                Some(pred) => pred,
            };
            match pred {
                // FIXME(const_generics): Make sure that `<'a, 'b, const N: &'a &'b u32>` is sound
                // if we ever support that
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                | ty::PredicateKind::DynCompatible(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::NormalizesTo(..)
                | ty::PredicateKind::AliasRelate(..) => {}

                // We need to search through *all* WellFormed predicates
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
                    wf_args.push(arg);
                }

                // We need to register region relationships
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(
                    ty::OutlivesPredicate(r_a, r_b),
                )) => outlives_bounds.push(ty::OutlivesPredicate(r_a.into(), r_b)),

                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    ty_a,
                    r_b,
                ))) => outlives_bounds.push(ty::OutlivesPredicate(ty_a.into(), r_b)),
            }
        }
    }

    // This call to `select_all_or_error` is necessary to constrain inference variables, which we
    // use further down when computing the implied bounds.
    match ocx.select_all_or_error().as_slice() {
        [] => (),
        _ => return Err(NoSolution),
    }

    // We lazily compute the outlives components as
    // `select_all_or_error` constrains inference variables.
    let mut implied_bounds = Vec::new();
    for ty::OutlivesPredicate(a, r_b) in outlives_bounds {
        match a.unpack() {
            ty::GenericArgKind::Lifetime(r_a) => {
                implied_bounds.push(OutlivesBound::RegionSubRegion(r_b, r_a))
            }
            ty::GenericArgKind::Type(ty_a) => {
                let mut ty_a = ocx.infcx.resolve_vars_if_possible(ty_a);
                // Need to manually normalize in the new solver as `wf::obligations` does not.
                if ocx.infcx.next_trait_solver() {
                    ty_a = ocx
                        .deeply_normalize(&ObligationCause::dummy(), param_env, ty_a)
                        .map_err(|_| NoSolution)?;
                }
                let mut components = smallvec![];
                push_outlives_components(tcx, ty_a, &mut components);
                implied_bounds.extend(implied_bounds_from_components(r_b, components))
            }
            ty::GenericArgKind::Const(_) => {
                unreachable!("consts do not participate in outlives bounds")
            }
        }
    }

    Ok(implied_bounds)
}

/// When we have an implied bound that `T: 'a`, we can further break
/// this down to determine what relationships would have to hold for
/// `T: 'a` to hold. We get to assume that the caller has validated
/// those relationships.
fn implied_bounds_from_components<'tcx>(
    sub_region: ty::Region<'tcx>,
    sup_components: SmallVec<[Component<TyCtxt<'tcx>>; 4]>,
) -> Vec<OutlivesBound<'tcx>> {
    sup_components
        .into_iter()
        .filter_map(|component| {
            match component {
                Component::Region(r) => Some(OutlivesBound::RegionSubRegion(sub_region, r)),
                Component::Param(p) => Some(OutlivesBound::RegionSubParam(sub_region, p)),
                Component::Alias(p) => Some(OutlivesBound::RegionSubAlias(sub_region, p)),
                Component::Placeholder(_p) => {
                    // FIXME(non_lifetime_binders): Placeholders don't currently
                    // imply anything for outlives, though they could easily.
                    None
                }
                Component::EscapingAlias(_) =>
                // If the projection has escaping regions, don't
                // try to infer any implied bounds even for its
                // free components. This is conservative, because
                // the caller will still have to prove that those
                // free components outlive `sub_region`. But the
                // idea is that the WAY that the caller proves
                // that may change in the future and we want to
                // give ourselves room to get smarter here.
                {
                    None
                }
                Component::UnresolvedInferenceVariable(..) => None,
            }
        })
        .collect()
}
