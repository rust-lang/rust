//! Provider for the `implied_outlives_bounds` query.
//! Do not call this query directory. See
//! [`rustc_trait_selection::traits::query::type_op::implied_outlives_bounds`].

use rustc_infer::infer::canonical;
use rustc_infer::infer::outlives::components::{push_outlives_components, Component};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::query::OutlivesBound;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::{CanonicalTyGoal, Fallible, NoSolution};
use rustc_trait_selection::traits::wf;
use rustc_trait_selection::traits::ObligationCause;
use smallvec::{smallvec, SmallVec};

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { implied_outlives_bounds, ..*p };
}

fn implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalTyGoal<'tcx>,
) -> Fallible<canonical::CanonicalQueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>> {
    tcx.infer_ctxt().enter_canonical_trait_query(&goal, |ocx, key| {
        let (param_env, ty) = key.into_parts();

        compute_implied_outlives_bounds(tcx, param_env, ty, |ty| {
            let ty = ocx.normalize(&ObligationCause::dummy(), param_env, ty);
            if !ocx.select_all_or_error().is_empty() {
                return Err(NoSolution);
            }
            let ty = ocx.infcx.resolve_vars_if_possible(ty);
            assert!(!ty.has_non_region_infer());
            Ok(ty)
        })
    })
}

/// For the sake of completeness, we should be careful when dealing with inference artifacts:
/// - This function shouldn't access an InferCtxt.
/// - `ty` must be fully resolved.
/// - `normalize_op` must return a fully resolved type.
fn compute_implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
    normalize_op: impl Fn(Ty<'tcx>) -> Fallible<Ty<'tcx>>,
) -> Fallible<Vec<OutlivesBound<'tcx>>> {
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
        for obligation in wf::unnormalized_obligations(tcx, param_env, arg) {
            assert!(!obligation.has_escaping_bound_vars());
            let Some(pred) = obligation.predicate.kind().no_bound_vars() else {
                continue;
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
                ty::PredicateKind::WellFormed(arg) => wf_args.push(arg),

                // We need to register region relationships
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(
                    ty::OutlivesPredicate(r_a, r_b),
                )) => outlives_bounds.push(OutlivesBound::RegionSubRegion(r_b, r_a)),

                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(
                    ty_a,
                    r_b,
                ))) => {
                    let ty_a = normalize_op(ty_a)?;
                    let mut components = smallvec![];
                    push_outlives_components(tcx, ty_a, &mut components);
                    outlives_bounds.extend(implied_bounds_from_components(r_b, components))
                }
            }
        }
    }

    Ok(outlives_bounds)
}

/// When we have an implied bound that `T: 'a`, we can further break
/// this down to determine what relationships would have to hold for
/// `T: 'a` to hold. We get to assume that the caller has validated
/// those relationships.
fn implied_bounds_from_components<'tcx>(
    sub_region: ty::Region<'tcx>,
    sup_components: SmallVec<[Component<'tcx>; 4]>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> {
    sup_components.into_iter().filter_map(move |component| {
        match component {
            Component::Region(r) => Some(OutlivesBound::RegionSubRegion(sub_region, r)),
            Component::Param(p) => Some(OutlivesBound::RegionSubParam(sub_region, p)),
            Component::Alias(p) => Some(OutlivesBound::RegionSubAlias(sub_region, p)),
            // If the projection has escaping regions, don't
            // try to infer any implied bounds even for its
            // free components. This is conservative, because
            // the caller will still have to prove that those
            // free components outlive `sub_region`. But the
            // idea is that the WAY that the caller proves
            // that may change in the future and we want to
            // give ourselves room to get smarter here.
            Component::EscapingAlias(_) => None,
            Component::UnresolvedInferenceVariable(..) => bug!("inference var in implied bounds"),
        }
    })
}
