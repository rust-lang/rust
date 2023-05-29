use crate::traits::query::NoSolution;
use crate::traits::wf;
use crate::traits::ObligationCtxt;

use rustc_infer::infer::canonical::Canonical;
use rustc_infer::infer::outlives::components::{push_outlives_components, Component};
use rustc_infer::traits::query::OutlivesBound;
use rustc_middle::infer::canonical::CanonicalQueryResponse;
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::source_map::DUMMY_SP;
use smallvec::{smallvec, SmallVec};

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ImpliedOutlivesBounds<'tcx> {
    pub ty: Ty<'tcx>,
}

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
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution> {
        // FIXME this `unchecked_map` is only necessary because the
        // query is defined as taking a `ParamEnvAnd<Ty>`; it should
        // take an `ImpliedOutlivesBounds` instead
        let canonicalized = canonicalized.unchecked_map(|ParamEnvAnd { param_env, value }| {
            let ImpliedOutlivesBounds { ty } = value;
            param_env.and(ty)
        });

        tcx.implied_outlives_bounds(canonicalized)
    }

    fn perform_locally_in_new_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        compute_implied_outlives_bounds_inner(ocx, key.param_env, key.value.ty)
    }
}

pub fn compute_implied_outlives_bounds_inner<'tcx>(
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

    let mut outlives_bounds: Vec<ty::OutlivesPredicate<ty::GenericArg<'tcx>, ty::Region<'tcx>>> =
        vec![];

    while let Some(arg) = wf_args.pop() {
        if !checked_wf_args.insert(arg) {
            continue;
        }

        // Compute the obligations for `arg` to be well-formed. If `arg` is
        // an unresolved inference variable, just substituted an empty set
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
                    ty::PredicateKind::Clause(ty::Clause::Projection(..))
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
                ty::PredicateKind::Clause(ty::Clause::Trait(..))
                // FIXME(const_generics): Make sure that `<'a, 'b, const N: &'a &'b u32>` is sound
                // if we ever support that
                | ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::Clause::Projection(..))
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::AliasRelate(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => {}

                // We need to search through *all* WellFormed predicates
                ty::PredicateKind::WellFormed(arg) => {
                    wf_args.push(arg);
                }

                // We need to register region relationships
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(ty::OutlivesPredicate(
                    r_a,
                    r_b,
                ))) => outlives_bounds.push(ty::OutlivesPredicate(r_a.into(), r_b)),

                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(
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
    let implied_bounds = outlives_bounds
        .into_iter()
        .flat_map(|ty::OutlivesPredicate(a, r_b)| match a.unpack() {
            ty::GenericArgKind::Lifetime(r_a) => vec![OutlivesBound::RegionSubRegion(r_b, r_a)],
            ty::GenericArgKind::Type(ty_a) => {
                let ty_a = ocx.infcx.resolve_vars_if_possible(ty_a);
                let mut components = smallvec![];
                push_outlives_components(tcx, ty_a, &mut components);
                implied_bounds_from_components(r_b, components)
            }
            ty::GenericArgKind::Const(_) => unreachable!(),
        })
        .collect();

    Ok(implied_bounds)
}

/// When we have an implied bound that `T: 'a`, we can further break
/// this down to determine what relationships would have to hold for
/// `T: 'a` to hold. We get to assume that the caller has validated
/// those relationships.
fn implied_bounds_from_components<'tcx>(
    sub_region: ty::Region<'tcx>,
    sup_components: SmallVec<[Component<'tcx>; 4]>,
) -> Vec<OutlivesBound<'tcx>> {
    sup_components
        .into_iter()
        .filter_map(|component| {
            match component {
                Component::Region(r) => Some(OutlivesBound::RegionSubRegion(sub_region, r)),
                Component::Param(p) => Some(OutlivesBound::RegionSubParam(sub_region, p)),
                Component::Alias(p) => Some(OutlivesBound::RegionSubAlias(sub_region, p)),
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
