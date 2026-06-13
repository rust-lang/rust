use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Flags, ImplTraitInTraitData, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor,
};

use crate::infer::outlives::test_type_match;
use crate::infer::region_constraints::VerifyIfEq;
use crate::regions::{region_known_to_outlive, ty_known_to_outlive};

/// For a given alias type, this returns the set of (identity) generic args that
/// are relevant for liveness, that can be inferred from outlives bounds on the
/// alias itself, and the explicit and implicit outlives clauses of the alias.
/// Callers should instantiate the returned args with the concrete args of the alias.
///
/// There are three cases to consider:
/// 1. If there are *no* outlives bounds, then we return None.
/// 2. If there is a `'static` outlives bound, then we know that all args are
///    irrelevant, so we return an empty list.
/// 3. If there are *any* outlives bounds, then we find any args that are known
///    to outlive those bounds, since those are the args whose regions the
///    underlying type could capture.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn live_args_for_alias_from_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: ty::AliasTyKind<'tcx>,
) -> Option<ty::EarlyBinder<'tcx, Vec<ty::GenericArg<'tcx>>>> {
    let def_id = kind.def_id();
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

    // We first want to collect the outlives bounds of the alias.
    let bounds = tcx.item_bounds(def_id).instantiate_identity().skip_norm_wip();
    tracing::debug!(?bounds);
    let alias_ty = Ty::new_alias(tcx, ty::AliasTy::new_from_args(tcx, kind, self_identity_args));
    let outlives_regions: Vec<_> = bounds
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
    tracing::debug!(?outlives_regions);

    // If there are no outlives bounds, then all (non-bivariant) args are potentially live.
    if outlives_regions.is_empty() {
        return None;
    }

    // If any of the outlives bounds are `'static`, then we know the alias
    // doesn't capture *any* regions, so we can skip visiting any regions at all.
    //
    // I was originally a bit concerned about something like `'a: 'static`, and
    // whether or not we need to mark `'a` as live. I don't think *today* we do,
    // since I think regions that outlive `'static` are special enough, but I
    // *could* imagine some world where we need to be more careful about this.
    // Given I can't find a test that goes wrong, I'm going to leave in this optimization.
    if outlives_regions.contains(&tcx.lifetimes.re_static) {
        tracing::debug!("alias has a 'static outlives bound, so skipping visiting any regions");
        return Some(ty::EarlyBinder::bind(vec![]));
    }

    // Okay, so we know we have some outlives bounds, and that none of them are `'static`.
    // Now, we need to find all other potentially-live args, those that outlive
    // an outlives-bound region. `args_known_to_outlive_alias_params` does this
    // for us, and in the case of opaques only includes *captured* regions, too.

    let args_known_to_outlive =
        tcx.args_known_to_outlive_alias_params(def_id).as_ref().skip_binder();
    tracing::debug!(?args_known_to_outlive);
    let mut live_args: Option<FxIndexSet<ty::GenericArg<'tcx>>> = None;
    for outlives_region in outlives_regions {
        let Some(outlives_regions) =
            args_known_to_outlive.iter().find(|(r, _)| *r == outlives_region)
        else {
            continue;
        };
        let new_live_args = outlives_regions.1.iter().copied().collect();
        live_args = Some(match live_args.take() {
            None => new_live_args,
            Some(prev) => prev.intersection(&new_live_args).copied().collect(),
        });
    }
    live_args.map(|c| ty::EarlyBinder::bind(c.into_iter().collect()))
}

/// For each region param of this alias compute the identity args that are known
/// to outlive it, given only the alias's declared where-clauses.
///
/// Note: for opaques (including synthetic associated types from RPITITs),
/// the outlives relationships are identified in the context of the *parent*,
/// since bounds and well-formed types are not lowered.
// FIXME: this likely should return a `BitSet` instead of a `Vec<Vec<>>`
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_alias_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, Vec<(ty::Region<'tcx>, Vec<ty::GenericArg<'tcx>>)>> {
    match tcx.def_kind(def_id) {
        DefKind::OpaqueTy => args_known_to_outlive_opaque_params(tcx, def_id),
        DefKind::AssocTy
            if let Some(ImplTraitInTraitData::Trait { fn_def_id: _, opaque_def_id }) =
                tcx.opt_rpitit_info(def_id.to_def_id()) =>
        {
            args_known_to_outlive_opaque_params(tcx, opaque_def_id.expect_local())
        }
        DefKind::AssocTy | DefKind::TyAlias => {
            args_known_to_outlive_associated_type_params(tcx, def_id)
        }
        kind => {
            bug!("improper def_kind {kind:?} passed to `live_args_for_alias_from_outlives_bounds`")
        }
    }
}

#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_opaque_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, Vec<(ty::Region<'tcx>, Vec<ty::GenericArg<'tcx>>)>> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

    let mut result = Vec::new();

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
        rustc_hir::OpaqueTyOrigin::TyAlias { parent, .. } => (parent, FxIndexSet::default()),
    };
    let parent_param_env = tcx.param_env(parent_def_id);
    tracing::debug!(?parent_param_env);

    // Map the outlives regions to the parent regions
    let generics = tcx.generics_of(def_id);
    let opaque_captured_lifetimes = tcx.opaque_captured_lifetimes(def_id);
    let mut parent_outlives_regions = Vec::with_capacity(generics.own_params.len());
    for opaque_arg in self_identity_args[generics.parent_count..].iter() {
        let Some(opaque_region) = opaque_arg.as_region() else {
            continue;
        };
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
        parent_outlives_regions.push((parent_region, opaque_region));
    }
    tracing::debug!(?parent_outlives_regions);

    for (parent_outlived_region, opaque_outlived_region) in parent_outlives_regions.iter() {
        let mut opaque_outlives_args = Vec::with_capacity(self_identity_args.len());
        for parent_outlives_arg in self_identity_args[..generics.parent_count].iter() {
            let type_outlives = match parent_outlives_arg.kind() {
                // Consts don't have any non-static regions
                ty::GenericArgKind::Const(_) => continue,
                // Lifetimes should be captured
                ty::GenericArgKind::Lifetime(_) => continue,
                ty::GenericArgKind::Type(t) => ty_known_to_outlive(
                    tcx,
                    def_id,
                    parent_param_env,
                    &wf_tys,
                    t,
                    *parent_outlived_region,
                ),
            };
            if !type_outlives {
                continue;
            }

            // Types aren't captured, so don't need to map to the opaque
            opaque_outlives_args.push(*parent_outlives_arg);
        }

        for (_, opaque_lt) in opaque_captured_lifetimes.iter() {
            let parent_outlives_region = tcx.map_opaque_lifetime_to_parent_lifetime(*opaque_lt);

            let region_outlives = parent_outlives_region == *parent_outlived_region
                || region_known_to_outlive(
                    tcx,
                    def_id,
                    parent_param_env,
                    &wf_tys,
                    parent_outlives_region,
                    *parent_outlived_region,
                );
            if !region_outlives {
                continue;
            }

            let param = generics
                .own_params
                .iter()
                .find(|param| param.def_id == opaque_lt.to_def_id())
                .unwrap();
            let opaque_region =
                ty::Region::new_early_param(tcx, param.to_early_bound_region_data());

            opaque_outlives_args.push(opaque_region.into());
        }

        result.push((*opaque_outlived_region, opaque_outlives_args));
    }

    ty::EarlyBinder::bind(result)
}

#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_associated_type_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, Vec<(ty::Region<'tcx>, Vec<ty::GenericArg<'tcx>>)>> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
    let param_env = tcx.param_env(def_id);
    tracing::debug!(?param_env);
    let wf_tys = FxIndexSet::default();
    let mut result = Vec::new();
    for outlived_arg in self_identity_args.iter() {
        let Some(outlived_region) = outlived_arg.as_region() else {
            continue;
        };
        let outliving_args = self_identity_args
            .iter()
            .filter(|arg| match arg.kind() {
                ty::GenericArgKind::Lifetime(r) => {
                    region_known_to_outlive(tcx, def_id, param_env, &wf_tys, r, outlived_region)
                }
                ty::GenericArgKind::Type(t) => {
                    ty_known_to_outlive(tcx, def_id, param_env, &wf_tys, t, outlived_region)
                }
                ty::GenericArgKind::Const(_) => false,
            })
            .collect();
        result.push((outlived_region, outliving_args));
    }
    ty::EarlyBinder::bind(result)
}

/// For a param-env clause `for<'v..> <T as Trait>::Assoc<..>: 'bound` that
/// applies to `ty` (an alias with `alias_def_id`), returns the set of (identity) args
/// that the underlying type could possibly capture, as restricted by this clause.
///
/// As an example, let's imagine we had the following associated type definition:
/// ```ignore (illustrative)
/// type Assoc<'a, 'b, 'c: 'a> = (&'a &'c (), &'b ());
/// ```
///
/// the following clause:
/// ```ignore (illustrative)
/// for<'x, 'y> T::Assoc<'x, 'x, 'y>: 'x
/// ```
///
/// We know from the clause alone that *given some substitution of `T:Assoc`*,
/// we know that it can capture either the first or the second region. However,
/// the bounds on the associated type itself additionally imply that the
/// third region can *also* be captured, because it outlives the first.
///
/// Now, let's assume we had this clause:
/// ```ignore (illustrative)
/// for<'x, 'y> T::Assoc<'x, 'y, 'x>: 'x
/// ```
///
/// Here, we know that `'a` and `'c` could be captured, but there is no outlives
/// relationship to `'b` for either of those, so the underlying type can't
/// capture any arg containing `'b`.
///
/// Note: because higher-ranked bounds don't have implications, there will be
/// some cases (like `for<'x, 'y, 'z> T::Assoc<'x, 'y, 'z>: 'x`) that won't
/// be satisfiable today, but the logic here should hold whenever there *is*.
///
/// Returns `None` if the clause doesn't apply to `ty` or gives us no information.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
fn live_args_for_clause<'tcx>(
    tcx: TyCtxt<'tcx>,
    alias_def_id: DefId,
    ty: Ty<'tcx>,
    clause: ty::Clause<'tcx>,
) -> Option<FxIndexSet<ty::GenericArg<'tcx>>> {
    let outlives = clause.as_type_outlives_clause()?;
    let ty::Alias(ty::AliasTy { kind: clause_alias_kind, args: clause_args, .. }) =
        *outlives.skip_binder().0.kind()
    else {
        return None;
    };
    if clause_alias_kind.def_id() != alias_def_id {
        return None;
    }

    // Here, we're just using this to check if the clause *could apply* to `ty`,
    // but importantly we don't want to use the returned region, because that is
    // the "last visited" region in `ty` that matches the outlves bound. Actually,
    // we want *all* the identity regions in `ty` that match the outlives bound.
    test_type_match::extract_verify_if_eq(
        tcx,
        &outlives.map_bound(|ty::OutlivesPredicate(ty, bound)| VerifyIfEq { ty, bound }),
        ty,
    )?;

    let bound_region = outlives.skip_binder().1;
    let clause_identity_args = ty::GenericArgs::identity_for_item(tcx, alias_def_id);
    match bound_region.kind() {
        // The underlying type must outlive `'static`, so it can't capture any of the args at all.
        ty::ReStatic => Some(FxIndexSet::default()),
        ty::ReBound(_, br) => {
            // The bound is one of the clause's higher-ranked vars. Find the arg
            // positions it occupies, then (at the alias's identity level) find
            // all args that are known to outlive one of those positions given
            // the alias's declared bounds -- only those can be captured by the
            // underlying type.
            let mut bound_regions = Vec::new();
            for (clause_arg, identity_arg) in clause_args.iter().zip(clause_identity_args.iter()) {
                match clause_arg.kind() {
                    ty::GenericArgKind::Lifetime(r) => {
                        if let ty::ReBound(_, arg_br) = r.kind()
                            && arg_br.var == br.var
                        {
                            bound_regions.push(identity_arg.expect_region());
                        }
                    }
                    ty::GenericArgKind::Type(_) | ty::GenericArgKind::Const(_) => {
                        // A bound var inside a type or const arg (e.g.
                        // `for<'a> <F as FnOnce<(&'a mut i32,)>>::Output: 'a`)
                        // can't be reasoned about at the identity-param level,
                        // so conservatively treat the clause as giving no
                        // restriction at all.
                        if clause_arg.has_escaping_bound_vars() {
                            return None;
                        }
                    }
                }
            }
            if bound_regions.is_empty() {
                // The bound var doesn't appear in the args at all, so the clause
                // requires the underlying type to outlive *every* region, which
                // is equivalent to a `'static` bound.
                return Some(FxIndexSet::default());
            }

            // The underlying type can capture any arg that's known to outlive one
            // of the bound var's positions (they're all instantiated to the same
            // region at any use site this clause applies to).
            let args_known_to_outlive = tcx.args_known_to_outlive_alias_params(alias_def_id);
            tracing::debug!(?bound_regions, ?args_known_to_outlive);
            let mut capturable_args = FxIndexSet::default();
            for &bound_region in &bound_regions {
                let (_, outliving_args) = args_known_to_outlive
                    .as_ref()
                    .skip_binder()
                    .iter()
                    .find(|(region, _)| *region == bound_region)
                    .unwrap();
                capturable_args.extend(outliving_args.iter().copied());
            }
            Some(capturable_args)
        }
        // A free region (e.g. `for<'a> T::Assoc<'a, 'b>: 'b`): no higher-ranked
        // arg of the alias can be proven (by a caller) to outlive a free region
        // today, and pinned args only ever match universal regions (which are
        // always live), so we conservatively treat this as giving no
        // restriction.
        //
        // NB: if we ever get implied bounds inside binders, a bound var *could*
        // be assumed to outlive a free region (e.g. `for<'a> T::Assoc<'x, 'a>: 'x`
        // with a declared `'b: 'a` on the assoc type implies `'a: 'x` under the
        // binder), so this would need revisiting -- though the result would
        // still only need to include bound-var positions.
        _ => None,
    }
}

/// Visits free regions in the type that are relevant for liveness computation.
/// These regions are passed to `OP`.
///
/// Specifically, we visit all of the regions of types recursively, except if
/// the type is an alias, we look at the outlives bounds in the param-env and
/// the alias's item bounds. Each such bound restricts which of the alias's
/// args the underlying type could have captured, so only those (capturable)
/// args are visited. If there are no applicable bounds, we walk through the
/// alias's (non-bivariant) args structurally.
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

                // For aliases other than opaques, we have to consider two
                // sources of information to identity potentially-live args:
                // - Bounds on alias item itself
                // - Outlives clauses on the current function that apply to the alias
                //
                // Each source of information *restricts* the set of potentially-live
                // args independently: only the args that can be live for *every*
                // source of information can be actually live, so we take the intersection.
                let def_id = kind.def_id();
                let mut capturable: Option<FxIndexSet<ty::GenericArg<'tcx>>> = None;
                let mut restrict = |capturable_args: FxIndexSet<ty::GenericArg<'tcx>>| {
                    capturable = Some(match capturable.take() {
                        None => capturable_args,
                        Some(prev) => prev.intersection(&capturable_args).copied().collect(),
                    });
                };

                if let Some(live_args) = tcx.live_args_for_alias_from_outlives_bounds(kind) {
                    restrict(live_args.as_ref().skip_binder().iter().copied().collect());
                }

                for clause in param_env.caller_bounds() {
                    if let Some(capturable_args) = live_args_for_clause(tcx, def_id, ty, clause) {
                        restrict(capturable_args);
                    }
                }
                tracing::debug!(?capturable);

                match capturable {
                    Some(capturable_args) => {
                        for arg in capturable_args {
                            let arg =
                                ty::EarlyBinder::bind(arg).instantiate(tcx, args).skip_norm_wip();
                            arg.visit_with(self);
                        }
                    }
                    None => {
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
            }

            _ => ty.super_visit_with(self),
        }
    }
}
