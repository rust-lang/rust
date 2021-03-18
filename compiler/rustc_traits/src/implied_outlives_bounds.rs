//! Provider for the `implied_outlives_bounds` query.
//! Do not call this query directory. See
//! [`rustc_trait_selection::traits::query::type_op::implied_outlives_bounds`].

use rustc_hir as hir;
use rustc_infer::infer::canonical::{self, Canonical};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::TraitEngineExt as _;
use rustc_middle::ty::outlives::Component;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::source_map::DUMMY_SP;
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::outlives_bounds::OutlivesBound;
use rustc_trait_selection::traits::query::{CanonicalTyGoal, Fallible, NoSolution};
use rustc_trait_selection::traits::wf;
use rustc_trait_selection::traits::FulfillmentContext;
use rustc_trait_selection::traits::TraitEngine;
use smallvec::{smallvec, SmallVec};

crate fn provide(p: &mut Providers) {
    *p = Providers { implied_outlives_bounds, ..*p };
}

fn implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalTyGoal<'tcx>,
) -> Result<
    &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
    NoSolution,
> {
    tcx.infer_ctxt().enter_canonical_trait_query(&goal, |infcx, _fulfill_cx, key| {
        let (param_env, ty) = key.into_parts();
        compute_implied_outlives_bounds(&infcx, param_env, ty)
    })
}

fn compute_implied_outlives_bounds<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Fallible<Vec<OutlivesBound<'tcx>>> {
    let tcx = infcx.tcx;

    // Sometimes when we ask what it takes for T: WF, we get back that
    // U: WF is required; in that case, we push U onto this stack and
    // process it next. Currently (at least) these resulting
    // predicates are always guaranteed to be a subset of the original
    // type, so we need not fear non-termination.
    let mut wf_args = vec![ty.into()];

    let mut implied_bounds = vec![];

    let mut fulfill_cx = FulfillmentContext::new();

    while let Some(arg) = wf_args.pop() {
        // Compute the obligations for `arg` to be well-formed. If `arg` is
        // an unresolved inference variable, just substituted an empty set
        // -- because the return type here is going to be things we *add*
        // to the environment, it's always ok for this set to be smaller
        // than the ultimate set. (Note: normally there won't be
        // unresolved inference variables here anyway, but there might be
        // during typeck under some circumstances.)
        let obligations = wf::obligations(infcx, param_env, hir::CRATE_HIR_ID, 0, arg, DUMMY_SP)
            .unwrap_or_default();

        // N.B., all of these predicates *ought* to be easily proven
        // true. In fact, their correctness is (mostly) implied by
        // other parts of the program. However, in #42552, we had
        // an annoying scenario where:
        //
        // - Some `T::Foo` gets normalized, resulting in a
        //   variable `_1` and a `T: Trait<Foo=_1>` constraint
        //   (not sure why it couldn't immediately get
        //   solved). This result of `_1` got cached.
        // - These obligations were dropped on the floor here,
        //   rather than being registered.
        // - Then later we would get a request to normalize
        //   `T::Foo` which would result in `_1` being used from
        //   the cache, but hence without the `T: Trait<Foo=_1>`
        //   constraint. As a result, `_1` never gets resolved,
        //   and we get an ICE (in dropck).
        //
        // Therefore, we register any predicates involving
        // inference variables. We restrict ourselves to those
        // involving inference variables both for efficiency and
        // to avoids duplicate errors that otherwise show up.
        fulfill_cx.register_predicate_obligations(
            infcx,
            obligations.iter().filter(|o| o.predicate.has_infer_types_or_consts()).cloned(),
        );

        // From the full set of obligations, just filter down to the
        // region relationships.
        implied_bounds.extend(obligations.into_iter().flat_map(|obligation| {
            assert!(!obligation.has_escaping_bound_vars());
            match obligation.predicate.kind().no_bound_vars() {
                None => vec![],
                Some(pred) => match pred {
                    ty::PredicateKind::Trait(..)
                    | ty::PredicateKind::Subtype(..)
                    | ty::PredicateKind::Projection(..)
                    | ty::PredicateKind::ClosureKind(..)
                    | ty::PredicateKind::ObjectSafe(..)
                    | ty::PredicateKind::ConstEvaluatable(..)
                    | ty::PredicateKind::ConstEquate(..)
                    | ty::PredicateKind::TypeWellFormedFromEnv(..) => vec![],
                    ty::PredicateKind::WellFormed(arg) => {
                        wf_args.push(arg);
                        vec![]
                    }

                    ty::PredicateKind::RegionOutlives(ty::OutlivesPredicate(r_a, r_b)) => {
                        vec![OutlivesBound::RegionSubRegion(r_b, r_a)]
                    }

                    ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ty_a, r_b)) => {
                        let ty_a = infcx.resolve_vars_if_possible(ty_a);
                        let mut components = smallvec![];
                        tcx.push_outlives_components(ty_a, &mut components);
                        implied_bounds_from_components(r_b, components)
                    }
                },
            }
        }));
    }

    // Ensure that those obligations that we had to solve
    // get solved *here*.
    match fulfill_cx.select_all_or_error(infcx) {
        Ok(()) => Ok(implied_bounds),
        Err(_) => Err(NoSolution),
    }
}

/// When we have an implied bound that `T: 'a`, we can further break
/// this down to determine what relationships would have to hold for
/// `T: 'a` to hold. We get to assume that the caller has validated
/// those relationships.
fn implied_bounds_from_components(
    sub_region: ty::Region<'tcx>,
    sup_components: SmallVec<[Component<'tcx>; 4]>,
) -> Vec<OutlivesBound<'tcx>> {
    sup_components
        .into_iter()
        .filter_map(|component| {
            match component {
                Component::Region(r) => Some(OutlivesBound::RegionSubRegion(sub_region, r)),
                Component::Param(p) => Some(OutlivesBound::RegionSubParam(sub_region, p)),
                Component::Projection(p) => Some(OutlivesBound::RegionSubProjection(sub_region, p)),
                Component::EscapingProjection(_) =>
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
