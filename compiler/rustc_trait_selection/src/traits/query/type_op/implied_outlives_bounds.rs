use std::ops::ControlFlow;

use rustc_infer::infer::TypeOutlivesConstraint;
use rustc_infer::infer::canonical::CanonicalQueryInput;
use rustc_infer::traits::query::OutlivesBound;
use rustc_infer::traits::query::type_op::ImpliedOutlivesBounds;
use rustc_middle::infer::canonical::CanonicalQueryResponse;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::outlives::{Component, push_outlives_components};
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, TypeVisitable, TypeVisitor};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::{DUMMY_SP, Span, sym};
use smallvec::{SmallVec, smallvec};

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
        tcx.implied_outlives_bounds((canonicalized, false))
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        span: Span,
    ) -> Result<Self::QueryResponse, NoSolution> {
        compute_implied_outlives_bounds_inner(ocx, key.param_env, key.value.ty, span, false)
    }
}

pub fn compute_implied_outlives_bounds_inner<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    disable_implied_bounds_hack: bool,
) -> Result<Vec<OutlivesBound<'tcx>>, NoSolution> {
    let normalize_ty = |ty| -> Result<_, NoSolution> {
        // We must normalize the type so we can compute the right outlives components.
        // for example, if we have some constrained param type like `T: Trait<Out = U>`,
        // and we know that `&'a T::Out` is WF, then we want to imply `U: 'a`.
        let ty = ocx
            .deeply_normalize(&ObligationCause::dummy_with_span(span), param_env, ty)
            .map_err(|_| NoSolution)?;
        Ok(ty)
    };

    // Sometimes when we ask what it takes for T: WF, we get back that
    // U: WF is required; in that case, we push U onto this stack and
    // process it next. Because the resulting predicates aren't always
    // guaranteed to be a subset of the original type, so we need to store the
    // WF args we've computed in a set.
    let mut checked_wf_args = rustc_data_structures::fx::FxHashSet::default();
    let mut wf_args = vec![ty.into(), normalize_ty(ty)?.into()];

    let mut outlives_bounds: Vec<OutlivesBound<'tcx>> = vec![];

    while let Some(arg) = wf_args.pop() {
        if !checked_wf_args.insert(arg) {
            continue;
        }

        // From the full set of obligations, just filter down to the region relationships.
        for obligation in
            wf::unnormalized_obligations(ocx.infcx, param_env, arg, DUMMY_SP, CRATE_DEF_ID)
                .into_iter()
                .flatten()
        {
            let pred = ocx
                .deeply_normalize(
                    &ObligationCause::dummy_with_span(span),
                    param_env,
                    obligation.predicate,
                )
                .map_err(|_| NoSolution)?;
            let Some(pred) = pred.kind().no_bound_vars() else {
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
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                    wf_args.push(term);
                }

                // We need to register region relationships
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(
                    ty::OutlivesPredicate(r_a, r_b),
                )) => outlives_bounds.push(OutlivesBound::RegionSubRegion(r_b, r_a)),

                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    ty_a,
                    r_b,
                ))) => {
                    let mut components = smallvec![];
                    push_outlives_components(ocx.infcx.tcx, ty_a, &mut components);
                    outlives_bounds.extend(implied_bounds_from_components(r_b, components))
                }
            }
        }
    }

    // If we detect `bevy_ecs::*::ParamSet` in the WF args list (and `disable_implied_bounds_hack`
    // or `-Zno-implied-bounds-compat` are not set), then use the registered outlives obligations
    // as implied bounds.
    if !disable_implied_bounds_hack
        && !ocx.infcx.tcx.sess.opts.unstable_opts.no_implied_bounds_compat
        && ty.visit_with(&mut ContainsBevyParamSet { tcx: ocx.infcx.tcx }).is_break()
    {
        for TypeOutlivesConstraint { sup_type, sub_region, .. } in
            ocx.infcx.take_registered_region_obligations()
        {
            let mut components = smallvec![];
            push_outlives_components(ocx.infcx.tcx, sup_type, &mut components);
            outlives_bounds.extend(implied_bounds_from_components(sub_region, components));
        }
    }

    Ok(outlives_bounds)
}

struct ContainsBevyParamSet<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsBevyParamSet<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        // We only care to match `ParamSet<T>` or `&ParamSet<T>`.
        match t.kind() {
            ty::Adt(def, _) => {
                if self.tcx.item_name(def.did()) == sym::ParamSet
                    && self.tcx.crate_name(def.did().krate) == sym::bevy_ecs
                {
                    return ControlFlow::Break(());
                }
            }
            ty::Ref(_, ty, _) => ty.visit_with(self)?,
            _ => {}
        }

        ControlFlow::Continue(())
    }
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
