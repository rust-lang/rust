use rustc_data_structures::indexmap::IndexSet;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{
    self, Flags, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    Unnormalized,
};

use crate::infer::outlives::test_type_match;
use crate::infer::region_constraints::VerifyIfEq;
use crate::regions::region_known_to_outlive;

/// For a given opaque type, this returns the set of generic args that are relevant for liveness, that can be inferred
/// from outlives bounds on the opaque.
///
/// There are three cases to consider:
/// 1. If there are *no* outlives bounds, then we return None.
/// 2. If there is a `'static` outlives bound, then we know that all regions are irrelevant, so we return an empty list.
/// 3. If there are *any* outlives bounds, Then we find any args that outlive those bounds.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn live_regions_for_opaque_from_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<ty::EarlyBinder<'tcx, Vec<ty::Region<'tcx>>>> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

    // We first want to collect the outlives bounds of the opaque.
    let bounds = tcx.item_bounds(def_id).instantiate_identity().skip_norm_wip();
    tracing::debug!(?bounds);
    let alias_ty = Ty::new_opaque(tcx, def_id.to_def_id(), self_identity_args);
    let opaque_outlives_regions: Vec<_> = bounds
        .iter()
        .filter_map(|clause| {
            let outlives = clause.as_type_outlives_clause()?;
            if let Some(outlives) = outlives.no_bound_vars()
                && outlives.0 == alias_ty
            {
                Some(outlives.1)
            } else {
                test_type_match::extract_verify_if_eq(
                    tcx,
                    &outlives
                        .map_bound(|ty::OutlivesPredicate(ty, bound)| VerifyIfEq { ty, bound }),
                    alias_ty,
                )
            }
        })
        .collect();
    tracing::debug!(?opaque_outlives_regions);

    // If there are no outlives bounds, then all (non-bivariant) args are potentially live.
    if opaque_outlives_regions.is_empty() {
        return None;
    }

    // If any of the outlives bounds are `'static`, then we know the opaque doesn't capture
    // *any* regions, so we can skip visiting any regions at all.
    //
    // Thinking about it, I was originally a bit concerned about something like `'a: 'static`, and
    // whether or not we need to mark `'a` as live. I don't think *today* we do, since I think regions
    // that outlive `'static` are special enough, but I *could* imagine some world where we need to be
    // more careful about this. Given I can't find a test that goes wrong, I'm going to leave in this
    // optimization.
    if opaque_outlives_regions.contains(&tcx.lifetimes.re_static) {
        tracing::debug!("opaque has a 'static outlives bound, so skipping visiting any regions");
        return Some(ty::EarlyBinder::bind(vec![]));
    }

    // Okay, so we know we have some outlives bounds, and that none of them are `'static`.
    // Now, we need to find all other potentially-live regions,
    // those that outlive an outlives-bound region and are captured.
    // We will map both the opaque outlives regions and the set of captured regions
    // back to the parent, and then use all bounds (explicit and implied) to
    // find the set of captured regions that outlive the outlives bounds.

    let opaque_captured_lifetimes = tcx.opaque_captured_lifetimes(def_id);
    tracing::debug!(?opaque_captured_lifetimes);

    // Map the outlives regions to the parent regions
    let generics = tcx.generics_of(def_id);
    let parent_outlives_regions: Vec<_> = opaque_outlives_regions
        .iter()
        .map(|opaque_region| {
            let region_def_id = match opaque_region.kind() {
                ty::ReEarlyParam(ebr) => generics.param_at(ebr.index as usize, tcx).def_id,
                _ => panic!("unexpected region `{opaque_region}` in opaque bounds"),
            };
            let region_param =
                generics.own_params.iter().find(|param| param.def_id == region_def_id).unwrap();
            let (_, opaque_region_def_id) = opaque_captured_lifetimes
                .iter()
                .find(|(_, opaque_r)| opaque_r.to_def_id() == region_def_id)
                .unwrap();
            let parent_region = tcx.map_opaque_lifetime_to_parent_lifetime(*opaque_region_def_id);
            tracing::debug!(?region_def_id, ?region_param, ?parent_region);
            parent_region
        })
        .collect();
    tracing::debug!(?parent_outlives_regions);

    // Map the captured regions to the parent regions
    let mut parent_captured_regions: Vec<(ty::Region<'tcx>, LocalDefId)> =
        Vec::with_capacity(opaque_captured_lifetimes.len());
    for (_, opaque_lt) in opaque_captured_lifetimes.iter() {
        let parent_region = tcx.map_opaque_lifetime_to_parent_lifetime(*opaque_lt);
        parent_captured_regions.push((parent_region, *opaque_lt));
    }
    tracing::debug!(?parent_captured_regions);

    // For implied bounds, we need the set of WF types from the parents.
    //  - For functions, this is all the input and output types.
    //  - For type alias, there are no implied bounds, so this is empty. (FIXME: the alias type itself probably should be here?)
    let (parent_def_id, wf_tys) = match tcx.opaque_ty_origin(def_id) {
        rustc_hir::OpaqueTyOrigin::FnReturn { parent, .. }
        | rustc_hir::OpaqueTyOrigin::AsyncFn { parent, .. } => {
            let fn_sig = tcx.fn_sig(parent).instantiate_identity().skip_norm_wip();
            let (liberated_fn_sig, _) = tcx.instantiate_bound_regions(fn_sig, |br| {
                let kind = ty::LateParamRegionKind::from_bound(br.var, br.kind);
                ty::Region::new_late_param(tcx, parent, kind)
            });
            let wf_tys = liberated_fn_sig.inputs_and_output.iter().collect();
            (parent, wf_tys)
        }
        rustc_hir::OpaqueTyOrigin::TyAlias { parent, .. } => (parent, IndexSet::default()),
    };

    // Find all the opaque's captured regions that outlive the outlives bounds using the implied bounds
    let parent_param_env = tcx.param_env(parent_def_id);
    tracing::debug!(?parent_param_env);
    let mut opaque_live_regions = Vec::with_capacity(parent_outlives_regions.len());
    for (parent_captured_region, opaque_captured_def_id) in parent_captured_regions.iter() {
        let mut all_outlives = true;
        for parent_outlives_region in parent_outlives_regions.iter() {
            let known_outlives = region_known_to_outlive(
                tcx,
                parent_def_id.expect_local(),
                parent_param_env,
                &wf_tys,
                *parent_captured_region,
                *parent_outlives_region,
            );
            tracing::debug!(?parent_captured_region, ?parent_outlives_region, ?known_outlives);
            if !known_outlives {
                all_outlives = false;
                break;
            }
        }

        if all_outlives {
            let param = generics
                .own_params
                .iter()
                .find(|param| param.def_id == opaque_captured_def_id.to_def_id())
                .unwrap();
            let opaque_region =
                ty::Region::new_early_param(tcx, param.to_early_bound_region_data());
            opaque_live_regions.push(opaque_region);
        }
    }
    tracing::debug!(?opaque_live_regions);

    Some(ty::EarlyBinder::bind(opaque_live_regions))
}

/// Visits free regions in the type that are relevant for liveness computation.
/// These regions are passed to `OP`.
///
/// Specifically, we visit all of the regions of types recursively, except if
/// the type is an alias, we look at the outlives bounds in the param-env
/// and alias's item bounds. If there is a unique outlives bound, then visit
/// that instead. If there is not a unique but there is a `'static` outlives
/// bound, then don't visit anything. Otherwise, walk through the opaque's
/// regions structurally.
pub struct FreeRegionsVisitor<'tcx, OP: FnMut(ty::Region<'tcx>)> {
    pub tcx: TyCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub op: OP,
}

impl<'tcx, OP> TypeVisitor<TyCtxt<'tcx>> for FreeRegionsVisitor<'tcx, OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            // ignore bound regions, keep visiting
            ty::ReBound(_, _) => {}
            _ => (self.op)(r),
        }
    }

    #[tracing::instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        // We're only interested in types involving regions
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return;
        }

        // FIXME: Don't consider alias bounds on types that have escaping bound
        // vars. See #117455.
        if ty.has_escaping_bound_vars() {
            return ty.super_visit_with(self);
        }

        match *ty.kind() {
            // We can prove that an alias is live two ways:
            // 1. All the components are live.
            //
            // 2. There is a known outlives bound or where-clause, and that
            //    region is live.
            //
            // We search through the item bounds and where clauses for
            // either `'static` or a unique outlives region, and if one is
            // found, we just need to prove that that region is still live.
            // If one is not found, then we continue to walk through the alias.
            ty::Alias(ty::AliasTy { kind, args, .. }) => {
                let tcx = self.tcx;
                let param_env = self.param_env;

                // Opaques are special, because there are additional captured regions that we need to consider.
                if let ty::AliasTyKind::Opaque { def_id } = kind {
                    let opaque_outlives_args =
                        tcx.live_regions_for_opaque_from_outlives_bounds(def_id);

                    match opaque_outlives_args {
                        Some(opaque_live_regions) => {
                            for r in opaque_live_regions.as_ref().skip_binder() {
                                let r = ty::EarlyBinder::bind(*r)
                                    .instantiate(tcx, args)
                                    .skip_norm_wip();
                                r.visit_with(self);
                            }
                        }
                        None => {
                            let variances = tcx.variances_of(def_id);
                            for (idx, s) in args.iter().enumerate() {
                                if variances[idx] != ty::Bivariant {
                                    s.visit_with(self);
                                }
                            }
                        }
                    }

                    return;
                }

                let outlives_bounds: Vec<_> = tcx
                    .item_bounds(kind.def_id())
                    .iter_instantiated(tcx, args)
                    .map(Unnormalized::skip_norm_wip)
                    .chain(param_env.caller_bounds())
                    .filter_map(|clause| {
                        let outlives = clause.as_type_outlives_clause()?;
                        if let Some(outlives) = outlives.no_bound_vars()
                            && outlives.0 == ty
                        {
                            Some(outlives.1)
                        } else {
                            test_type_match::extract_verify_if_eq(
                                tcx,
                                &outlives.map_bound(|ty::OutlivesPredicate(ty, bound)| {
                                    VerifyIfEq { ty, bound }
                                }),
                                ty,
                            )
                        }
                    })
                    .collect();
                tracing::debug!(?outlives_bounds);
                // If we find `'static`, then we know the alias doesn't capture *any* regions.
                // Otherwise, all of the outlives regions should be equal -- if they're not,
                // we don't really know how to proceed, so we continue recursing through the
                // alias.
                if outlives_bounds.contains(&tcx.lifetimes.re_static) {
                    // no
                } else if let Some(r) = outlives_bounds.first()
                    && outlives_bounds[1..].iter().all(|other_r| other_r == r)
                {
                    assert!(r.type_flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS));
                    r.visit_with(self);
                } else {
                    // Skip lifetime parameters that are not captured, since they do
                    // not need to be live.
                    let variances = tcx.opt_alias_variances(kind);

                    for (idx, s) in args.iter().enumerate() {
                        if variances.map(|variances| variances[idx]) != Some(ty::Bivariant) {
                            s.visit_with(self);
                        }
                    }
                }
            }

            _ => ty.super_visit_with(self),
        }
    }
}
