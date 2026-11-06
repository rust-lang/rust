use std::debug_assert_matches;

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::region_constraints::GenericKind;
use rustc_infer::infer::{SubregionOrigin, TyCtxtInferExt, TypeOutlivesConstraint};
use rustc_middle::ty::{
    self, CallerBoundsIterator, Flags, ImplTraitInTraitData, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_span::{DUMMY_SP, bug};
use smallvec::SmallVec;

use crate::infer::outlives::test_type_match;
use crate::infer::region_constraints::VerifyIfEq;
use crate::regions::{
    OutlivesEnvironmentBuildExt, region_known_to_outlive, test_region_obligations,
};

/// Given a known `param_env` and a set of well formed types, can we prove that
/// `ty` outlives `region`.
///
/// Copied from `ty_known_to_outlive` without normalization for `ty` because we
/// don't want trait solving in liveness queries.
fn param_known_to_outlive<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,
    wf_tys: &FxIndexSet<Ty<'tcx>>,
    ty: Ty<'tcx>,
    region: ty::Region<'tcx>,
) -> bool {
    debug_assert_matches!(ty.kind(), ty::Param(_));

    test_region_obligations(tcx, id, param_env, wf_tys, |infcx| {
        infcx.register_type_outlives_constraint_inner(TypeOutlivesConstraint {
            sub_region: region,
            sup_type: ty,
            origin: SubregionOrigin::RelateParamBound(DUMMY_SP, ty, None),
        });
        Ok(())
    })
}

fn has_param_and_non_static_region<'tcx>(
    param: Ty<'tcx>,
    value: impl TypeVisitable<TyCtxt<'tcx>>,
) -> bool {
    struct ContainsParamAndRegionVisitor<'tcx> {
        param: Ty<'tcx>,
        contains_param: bool,
        has_region: bool,
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsParamAndRegionVisitor<'tcx> {
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if t == self.param {
                self.contains_param = true;
            }
            t.super_visit_with(self);
        }

        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            if !matches!(r.kind(), ty::ReStatic) {
                self.has_region = true;
            }
        }
    }

    let mut visitor =
        ContainsParamAndRegionVisitor { param, contains_param: false, has_region: false };
    value.visit_with(&mut visitor);
    visitor.contains_param && visitor.has_region
}

/// Okay, so the logic here is pretty "oh, I guess that works". The core idea is
/// that we need to know if a param can end up in an alias (and therefore must be
/// live). *Conservatively*, we could say that any *captured* param must be live,
/// but that is imprecise and results in some breakage (see the
/// `rpit-consumed-iterator-liveness` test, for example).
///
/// The simple case (handled elsewhere) is if the param outlives the *outlives
/// bounds* of the alias itself. In that case, then it is obviously *okay* for
/// it to end up in the hidden type and therefore must be live.
///
/// However, there is a more complicated case: aliases (and namely, projections).
/// If we have a projection like `<T as Trait>::Assoc`, we can have a where
/// clause like `T::Assoc: 'a`; then, that projection can end up in the hidden
/// type of an alias that outlives `'a`. And because of that, `T` must be live,
/// since loans within `T` can end up in the projection (and we don't have a
/// way to say that `T::Assoc` is live directly):
///
/// ```ignore (illustrative)
/// trait Trait { type Assoc; }
/// impl<'q> Trait for &'q u8 { type Assoc = &'q u8; }
/// ```
///
/// Naively, we could try to try to enumerate through all projections reachable
/// through `T` and ask if we know if that projection outlives the alias bounds,
/// and then mark `T` as live. That seems...difficult to get right. Instead, we
/// do something a little bit more clever, but also less precise.
///
/// The first thing to recognize: a bound like `T::Assoc: 'static` *does not*
/// mean that `T` must be live (even though that `T` lives longer than than some
/// defined alias outlives bound), because `T::Assoc` cannot carry any loans). So,
/// we can ignore `'static`.
///
/// Second, while we *could* further restrict lifetimes we care about to only
/// those that *actually* be captured in the alias (like we do for captured
/// regions), we can be a bit imprecise and just consider if the param outlives
/// *any* non-static region.
///
/// Next, we can divide the cases we need to consider into three cases:
/// 1) we can deduce that a projection from `T` outlives some lifetime `'a` from
///    the ParamEnv directly (either because we have something like `T::Assoc:'a`
///    in the bounds exactly, or this is implied from something like a
///    `Struct<T::Assoc>: 'a` bound).
/// 2) we can deduce that a projection from `T` outlives some lifetime `'a` from
///    implied bounds from WF tys (you can think of this as a special case of the
///    above)
/// 3) some *unknown* projection from `T` outlives some lifetime `'a` -
///    importantly, we know that projection must come from some some trait bound
///    that *is* in the ParamEnv (even if indirectly). For example:
///
/// ```ignore (illustrative)
/// trait Trait<'a> {
///   type Assoc1: 'a; // Lifetime is present in the trait signature
///   type Assoc2: 'static; // Contains no loans
///   type Assoc3<'a>: 'a; // Present in alias signature
/// }
///
/// trait Outer<'a> { type Assoc: Inner<'a>; }
/// trait Inner<'a> { type Assoc: 'a; }
/// ```
///
/// So, if we have a bound like `T: Trait<'a>`, then that can be used to deduce
/// that `<T as Trait<'a>>::Assoc1: 'a`. We could do that by enumerating the
/// type aliases on `Trait`, but we would need to be recursive (for the
/// `Inner/Outer` example above) and that's not fun. You can also possibly
/// imagine that `T` doesn't directly have to implement a trait that contains the
/// alias outliving `'a` (though, I haven't tried really to come up with an
/// example here).
///
/// So instead, we have a simple heuristic: if we have a trait bound that
/// mentions both the param we're asking about and *any* non-static lifetime,
/// then conservatively treat the param as live. We're actually a bit more
/// conservative, see below comments for details.
fn param_projection_can_outlive_non_static_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    env: &OutlivesEnvironment<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    // This is for case (3)
    param_env.caller_bounds().all_clauses(tcx).iter().any(|clause| {
        match clause.kind().skip_binder() {
        // The param outliving the pertinent alias bounds is already handled
        // elsewhere more precisely and anything else is not relevant.
        ty::ClauseKind::TypeOutlives(ty::OutlivesClause(subject, _)) if subject == ty => false,
        // No param here
        ty::ClauseKind::RegionOutlives(_) => false,
        // Covers things like `Struct<T::Assoc>: 'a` (though, this would get
        // covered by the `OutlivesEnvironment` below)
        ty::ClauseKind::TypeOutlives(_)
        | ty::ClauseKind::Trait(_)
        // Something like `<T as Trait>::Assoc<'a> = &'a u8`
        | ty::ClauseKind::Projection(_)
        // Needed to support const params
        | ty::ClauseKind::ConstArgHasType(..) => has_param_and_non_static_region(ty, clause),

        // In practice, I don't think these really come up, but we can be conservative.
        ty::ClauseKind::WellFormed(..)
        | ty::ClauseKind::ConstEvaluatable(..)
        | ty::ClauseKind::HostEffect(..)
        | ty::ClauseKind::UnstableFeature(..) => true,
    }
    }) || env.region_bound_pairs().iter().any(|bound| {
        // This covers cases (1) and (2) generally; while the above
        // `TypeOutlives` arm covers ParamEnv clauses, it does not cover WF tys
        if let GenericKind::Alias(alias) = bound.0 {
            has_param_and_non_static_region(ty, (alias, bound.1))
        } else {
            false
        }
    })
}

/// This is more a technical limitation than a logical one. In the implied
/// bounds computation we replace universal regions with region vars. So if we
/// equate regions during implied bounds computation (see the
/// `equality-implied-liveness` test), then we can lose known constraints. We
/// shouldn't need this if/when we change hir implied bounds computation to not
/// replace universal regions with region vars (as mir implied bounds does).
fn extend_wf_tys<'tcx>(wf_tys: &mut FxIndexSet<Ty<'tcx>>) {
    for ty in wf_tys.clone() {
        let mut walker = ty.walk();
        while let Some(arg) = walker.next() {
            let Some(subty) = arg.as_type() else { continue };
            if subty.has_escaping_bound_vars() {
                walker.skip_current_subtree();
                continue;
            }
            wf_tys.insert(subty);
        }
    }
}

#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn captured_args_for_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> DenseBitSet<u32> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
    let generics = tcx.generics_of(def_id);
    let mut live_args = DenseBitSet::new_empty(self_identity_args.len());

    // All of the opaques *own* args are obviously captured, so mark them as live
    live_args.insert_range((generics.parent_count as u32)..(self_identity_args.len() as u32));

    for opaque_arg in self_identity_args.iter().skip(generics.parent_count) {
        let Some(opaque_region) = opaque_arg.as_region() else {
            continue;
        };
        // The *parent* arg for a captured opaque lifetime is also live (although it is bivariant).
        match opaque_region.kind() {
            ty::ReEarlyParam(ebr) => {
                let region_def_id = generics.param_at(ebr.index as usize, tcx).def_id;
                let parent_region =
                    tcx.map_opaque_lifetime_to_parent_lifetime(region_def_id.expect_local());
                if let ty::ReEarlyParam(ebr) = parent_region.kind() {
                    live_args.insert(ebr.index);
                }
            }
            _ => panic!("unexpected region `{opaque_region}` in opaque bounds"),
        };
    }

    for (parent_idx, parent_arg) in
        self_identity_args.as_slice()[..generics.parent_count].iter().enumerate()
    {
        match parent_arg.kind() {
            // Lifetimes may or may not be captured; we handle these above
            ty::GenericArgKind::Lifetime(_) => continue,
            // Consts and types are always considered live
            ty::GenericArgKind::Const(_) | ty::GenericArgKind::Type(_) => {
                live_args.insert(parent_idx as u32);
            }
        }
    }

    live_args
}

/// For a given alias type, this returns the set of indices into the identity generic args that
/// are relevant for liveness, that can be inferred from outlives bounds on the
/// alias itself, and the explicit and implicit outlives clauses of the alias.
/// Callers should use the indices with the concrete args of the alias.
///
/// There are three cases to consider:
/// 1. If there are *no* outlives bounds, then all args are potentially live.
/// 2. If there is a `'static` outlives bound, then we know that all args are
///    irrelevant, so we return an empty set.
/// 3. If there are *any* outlives bounds, then we find any args that are known
///    to outlive those bounds, since those are the args whose regions the
///    underlying type could capture.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn live_args_for_alias_from_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: ty::AliasTyKind<'tcx>,
) -> DenseBitSet<u32> {
    let def_id = match kind {
        ty::AliasTyKind::Projection { def_id }
        | ty::AliasTyKind::Inherent { def_id }
        | ty::AliasTyKind::Opaque { def_id }
        | ty::AliasTyKind::Free { def_id } => def_id,
    };
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

    // We first want to collect the outlives bounds of the alias.
    let bounds = tcx.item_bounds(def_id).instantiate_identity().skip_norm_wip();
    tracing::debug!(?bounds);
    let alias_ty = Ty::new_alias(
        tcx,
        ty::IsRigid::No,
        ty::AliasTy::new_from_args(tcx, kind, self_identity_args),
    );
    let outlives_regions: Vec<_> = bounds
        .iter()
        .filter_map(|clause| {
            let ty::OutlivesClause(ty, region) = clause.as_type_outlives_clause()?.skip_binder();
            if ty != alias_ty {
                return None;
            }

            // Opaques can't have higher-ranked outlives item bounds. Higher-ranked item bounds
            // for GATs are instantiated with the GAT identity params, so the alias doesn't
            // contain any bound regions. If the region is bound, the alias outlives everything.
            // For example: `for<'a> Self::Assoc<'non_bound>: 'a`.
            if region.is_bound() { Some(tcx.lifetimes.re_static) } else { Some(region) }
        })
        .collect();
    tracing::debug!(?outlives_regions);

    // If there are no outlives bounds, then all captured args are potentially live.
    if outlives_regions.is_empty() {
        return match tcx.def_kind(def_id) {
            DefKind::OpaqueTy => tcx.captured_args_for_opaque(def_id).clone(),
            DefKind::AssocTy
                if let Some(ImplTraitInTraitData::Trait { fn_def_id: _, opaque_def_id }) =
                    tcx.opt_rpitit_info(def_id) =>
            {
                tcx.captured_args_for_opaque(opaque_def_id).clone()
            }
            DefKind::AssocTy | DefKind::TyAlias => {
                DenseBitSet::new_filled(self_identity_args.len())
            }
            kind => {
                bug!(
                    "improper def_kind {kind:?} passed to `live_args_for_alias_from_outlives_bounds`"
                )
            }
        };
    }

    // If any of the outlives bounds are `'static`, then we know the alias
    // doesn't capture *any* regions, so we can skip visiting any regions at all.
    //
    // I was originally a bit concerned about something like `'a: 'static`, and
    // whether or not we need to mark `'a` as live. I don't think that we do.
    //
    // To dig in a bit: Think about the function using this alias. For the alias
    // to be well-formed, then it must be proven that the arg (`'a` in this case)
    // outlives `'static`. Well, if that is proven *once*, then it must be true
    // across the entire function (because `'static` is free).
    //
    // I think this similarly applies to any other free region, like `'a: 'b`
    // where `'b` is *also* free. Though, we of course can't know *here* which
    // regions are going to be instantiated with free regions.
    if outlives_regions.contains(&tcx.lifetimes.re_static) {
        tracing::debug!("alias has a 'static outlives bound, so skipping visiting any regions");
        return DenseBitSet::new_empty(self_identity_args.len());
    }

    // Okay, so we know we have some outlives bounds, and that none of them are `'static`.
    // Now, we need to find all other potentially-live args, those that outlive
    // an outlives-bound region. `args_known_to_outlive_alias_params` does this
    // for us, and in the case of opaques only includes *captured* regions, too.

    let args_known_to_outlive = tcx.args_known_to_outlive_alias_params(def_id);
    tracing::debug!(?args_known_to_outlive);
    let mut live_args = DenseBitSet::new_filled(self_identity_args.len());
    let mut bound_args = DenseBitSet::new_empty(self_identity_args.len());
    for outlives_region in outlives_regions {
        let Some(outlives_params) = args_known_to_outlive
            .iter()
            .find(|(idx, _)| self_identity_args[*idx].as_region() == Some(outlives_region))
        else {
            continue;
        };
        live_args.intersect(&outlives_params.1);
        bound_args.insert(outlives_params.0 as u32);
    }
    // The outlives lifetimes themselves must always be considered live.
    live_args.union(&bound_args);
    live_args
}

/// For each region param of this alias compute the indices of the identity args
/// that are known to outlive it, given only the alias's declared where-clauses.
///
/// Note: for opaques (including synthetic associated types from RPITITs),
/// the outlives relationships are identified in the context of the *parent*,
/// since bounds and well-formed types are not lowered.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_alias_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Vec<(usize, DenseBitSet<u32>)> {
    match tcx.def_kind(def_id) {
        DefKind::OpaqueTy => args_known_to_outlive_opaque_params(tcx, def_id),
        DefKind::AssocTy
            if let Some(ImplTraitInTraitData::Trait { fn_def_id: _, opaque_def_id }) =
                tcx.opt_rpitit_info(def_id.to_def_id()) =>
        {
            args_known_to_outlive_opaque_params(tcx, opaque_def_id.expect_local())
        }
        DefKind::AssocTy | DefKind::TyAlias => args_known_to_outlive_non_opaque_params(tcx, def_id),
        kind => {
            bug!("improper def_kind {kind:?} passed to `live_args_for_alias_from_outlives_bounds`")
        }
    }
}

/// For each *captured* region of this alias, compute the *captured* identity
/// args that are known to outlive it, given the definition of the opaque type
/// in the the *parent* context.
///
/// Some examples:
/// ```ignore (illustrative)
/// // Returns `[('a, ['a]), ('b, ['b])]`
/// fn foo<'a, 'b>() -> impl Sized + use<'a, 'b> {}
///
/// // Returns `[('a, ['a]), ('b, ['b])]`
/// fn foo<'a: 'a, 'b>() -> impl Sized + use<'a, 'b> {}
///
/// // Returns `[('a, ['a])]`
/// fn foo<'a, 'b>() -> impl Sized + use<'a> {}
///
/// // Returns `[('a, ['a, 'b]), ('b, ['b])]`
/// fn foo<'a, 'b: 'a>() -> impl Sized + use<'a, 'b> {}
///
/// // Returns `[('a, ['a, 'b]), ('b, ['b])]`
/// fn foo<'a, 'b>(_: &'a &'b ()) -> impl Sized + use<'a, 'b> {}
/// ```
///
/// Importantly:
///   - *All* captured regions are considered (not just those in outlives bounds)
///   - It doesn't matter if the captured region is early-bound or late-bound
#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_opaque_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Vec<(usize, DenseBitSet<u32>)> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);

    let mut result = Vec::new();

    // For implied bounds, we need the set of WF types from the parents.
    //  - For functions, this is all the input and output types.
    //  - For type alias, there are no implied bounds, so this is empty.
    let (parent_def_id, mut wf_tys) = match tcx.opaque_ty_origin(def_id) {
        rustc_hir::OpaqueTyOrigin::FnReturn { parent, .. }
        | rustc_hir::OpaqueTyOrigin::AsyncFn { parent, .. }
        | rustc_hir::OpaqueTyOrigin::TyAlias { parent, .. } => {
            let wf_tys = FxIndexSet::from_iter(
                tcx.assumed_wf_types(parent.expect_local()).iter().map(|(ty, _)| *ty),
            );
            (parent, wf_tys)
        }
    };
    extend_wf_tys(&mut wf_tys);
    let parent_param_env = tcx.param_env(parent_def_id);
    tracing::debug!(?parent_param_env);

    // Map the outlives regions to the parent regions.
    // If we have `fn foo<'a>() -> impl Sized + 'a`, then this gets lowered as
    // ```ignore (illustrative)
    // opaque foo_opaque<'a0>: Sized + 'a0;
    // fn foo<'a>() -> foo::<'a>::foo_opaque<'a> { ... }
    // ```
    // This maps `'a0` to `'a`, because that is what will be used to get the
    // explicit and implied outlives relations.
    //
    // I suppose, an alternative way to do this would be iterate through all the
    // *parent* regions and then find those that are captured. This should be
    // basically equivalent (except with the added frustration of needing to
    // build a `Region` from the opaque region's `LocalDefId`).
    let generics = tcx.generics_of(def_id);
    let mut parent_outlives_regions = Vec::with_capacity(generics.own_params.len());
    for (opaque_arg_idx, opaque_arg) in
        self_identity_args.iter().enumerate().skip(generics.parent_count)
    {
        let Some(opaque_region) = opaque_arg.as_region() else {
            continue;
        };
        let region_def_id = match opaque_region.kind() {
            ty::ReEarlyParam(ebr) => generics.param_at(ebr.index as usize, tcx).def_id,
            _ => panic!("unexpected region `{opaque_region}` in opaque bounds"),
        };
        let parent_region =
            tcx.map_opaque_lifetime_to_parent_lifetime(region_def_id.expect_local());
        tracing::debug!(?region_def_id, ?parent_region);
        parent_outlives_regions.push((parent_region, opaque_arg_idx));
    }
    tracing::debug!(?parent_outlives_regions);
    if parent_outlives_regions.is_empty() {
        return result;
    }
    let infcx = tcx.infer_ctxt().build(ty::TypingMode::non_body_analysis());
    let outlives_env =
        OutlivesEnvironment::new(&infcx, def_id, parent_param_env, wf_tys.iter().copied());

    // Compute once since these don't depend on the outlives regions
    let parent_type_params: Vec<_> = self_identity_args.as_slice()[..generics.parent_count]
        .iter()
        .enumerate()
        .filter_map(|(idx, arg)| {
            let ty = arg.as_type()?;
            let projection_can_outlive = param_projection_can_outlive_non_static_region(
                tcx,
                parent_param_env,
                &outlives_env,
                ty,
            );
            Some((idx, ty, projection_can_outlive))
        })
        .collect();

    // For every captured region, we want to consider outlived args from two sources:
    // 1) *Types*: These come from *parent* generics (and are not duplicated to the opaque)
    // 2) *Captured Regions*
    //
    // In both cases, we need to check known outlives for the *parent* region, because that's where the param_env and wf_tys are.
    for (parent_outlived_region, opaque_outlived_arg_idx) in parent_outlives_regions.iter() {
        let mut opaque_outlives_args = DenseBitSet::new_empty(self_identity_args.len());
        for &(parent_outlived_arg_idx, ty, projection_can_outlive) in &parent_type_params {
            let type_outlives = projection_can_outlive
                || param_known_to_outlive(
                    tcx,
                    def_id,
                    parent_param_env,
                    &wf_tys,
                    ty,
                    *parent_outlived_region,
                );
            if !type_outlives {
                continue;
            }

            // Types aren't captured, so don't need to map to the opaque
            opaque_outlives_args.insert(parent_outlived_arg_idx as u32);
        }

        for &(parent_outlives_region, opaque_arg_idx) in parent_outlives_regions.iter() {
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

            opaque_outlives_args.insert(opaque_arg_idx as u32);

            // Although the parent arg is bivariant, it still participates in
            // (invariant) outlives relationships. So, we include it in the set
            // of outlives args.
            if let ty::ReEarlyParam(ebr) = parent_outlives_region.kind() {
                opaque_outlives_args.insert(ebr.index);
            }
        }

        result.push((*opaque_outlived_arg_idx, opaque_outlives_args));
    }

    result
}

#[tracing::instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn args_known_to_outlive_non_opaque_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Vec<(usize, DenseBitSet<u32>)> {
    let self_identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
    let param_env = tcx.param_env(def_id);
    tracing::debug!(?param_env);
    if !self_identity_args.iter().any(|arg| arg.as_region().is_some()) {
        return Vec::new();
    }
    let infcx = tcx.infer_ctxt().build(ty::TypingMode::non_body_analysis());
    let mut wf_tys: FxIndexSet<_> =
        tcx.assumed_wf_types(def_id).iter().map(|(ty, _)| *ty).collect();
    extend_wf_tys(&mut wf_tys);
    let outlives_env = OutlivesEnvironment::new(&infcx, def_id, param_env, wf_tys.iter().copied());

    // Compute this once per type parameter, rather than once per region.
    let mut projection_outlives_args = DenseBitSet::new_empty(self_identity_args.len());
    for (idx, arg) in self_identity_args.iter().enumerate() {
        if let Some(ty) = arg.as_type()
            && param_projection_can_outlive_non_static_region(tcx, param_env, &outlives_env, ty)
        {
            projection_outlives_args.insert(idx as u32);
        }
    }

    let mut result = Vec::new();
    for (outlived_arg_idx, outlived_arg) in self_identity_args.iter().enumerate() {
        let Some(outlived_region) = outlived_arg.as_region() else {
            continue;
        };
        let mut outliving_args = DenseBitSet::new_empty(self_identity_args.len());
        for (arg_idx, arg) in self_identity_args.iter().enumerate() {
            let outlives = match arg.kind() {
                ty::GenericArgKind::Lifetime(r) => {
                    region_known_to_outlive(tcx, def_id, param_env, &wf_tys, r, outlived_region)
                }
                ty::GenericArgKind::Type(t) => {
                    projection_outlives_args.contains(arg_idx as u32)
                        || param_known_to_outlive(
                            tcx,
                            def_id,
                            param_env,
                            &wf_tys,
                            t,
                            outlived_region,
                        )
                }
                ty::GenericArgKind::Const(_) => false,
            };
            if outlives {
                outliving_args.insert(arg_idx as u32);
            }
        }
        result.push((outlived_arg_idx, outliving_args));
    }
    result
}

/// For a param-env clause `for<'v..> <T as Trait>::Assoc<..>: 'bound` that
/// applies to `ty` (an alias with `alias_def_id`), returns the set of indices into the
/// identity args that the underlying type could possibly capture, as restricted by this clause.
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
/// Returns a filled set if the clause doesn't apply to `ty` or gives us no
/// information.
#[tracing::instrument(level = "debug", skip(tcx), ret)]
fn live_args_for_outlives_clause<'tcx>(
    tcx: TyCtxt<'tcx>,
    alias_def_id: DefId,
    ty: Ty<'tcx>,
    outlives: ty::Binder<'tcx, ty::TypeOutlivesClause<'tcx>>,
) -> DenseBitSet<u32> {
    let clause_identity_args = ty::GenericArgs::identity_for_item(tcx, alias_def_id);
    let no_restriction = || DenseBitSet::new_filled(clause_identity_args.len());

    // N.B. it's okay to skip the binder here (and in the rest of the function),
    // because all variables under binders do not escape
    let ty::Alias(_, ty::AliasTy { kind: clause_alias_kind, args: clause_args, .. }) =
        *outlives.skip_binder().0.kind()
    else {
        return no_restriction();
    };
    let clause_def_id = match clause_alias_kind {
        ty::AliasTyKind::Projection { def_id }
        | ty::AliasTyKind::Inherent { def_id }
        | ty::AliasTyKind::Opaque { def_id }
        | ty::AliasTyKind::Free { def_id } => def_id,
    };
    if clause_def_id != alias_def_id {
        return no_restriction();
    }

    // Here, we're just using this to check if the clause *could apply* to `ty`,
    // but importantly we don't want to use the returned region, because that is
    // the "last visited" region in `ty` that matches the outlves bound. Actually,
    // we want *all* the identity regions in `ty` that match the outlives bound.
    let Some(_) = test_type_match::extract_verify_if_eq(
        tcx,
        &outlives.map_bound(|ty::OutlivesClause(ty, bound)| VerifyIfEq { ty, bound }),
        ty,
    ) else {
        return no_restriction();
    };

    let outlived_region = outlives.skip_binder().1;
    match outlived_region.kind() {
        // The underlying type must outlive `'static`, so it can't capture any of the args at all.
        //
        // Of course, you may ask: "what if the function has a `'a: 'static` bound?" See the corresponding
        // comment in `live_args_for_alias_from_outlives_bounds` for why we don't need to worry about that.
        ty::ReStatic => DenseBitSet::new_empty(clause_identity_args.len()),
        ty::ReBound(_, br) => {
            // The bound is one of the clause's higher-ranked vars. Find the arg
            // positions it occupies, then (at the alias's identity level) find
            // all args that are known to outlive one of those positions given
            // the alias's declared bounds -- only those can be captured by the
            // underlying type.
            let mut outlived_regions = Vec::new();
            for (clause_arg, (identity_arg_idx, _identity_arg)) in
                clause_args.iter().zip(clause_identity_args.iter().enumerate())
            {
                match clause_arg.kind() {
                    ty::GenericArgKind::Lifetime(r) => {
                        if let ty::ReBound(_, arg_br) = r.kind()
                            && arg_br.var == br.var
                        {
                            outlived_regions.push(identity_arg_idx);
                        }
                    }
                    ty::GenericArgKind::Type(_) | ty::GenericArgKind::Const(_) => {
                        // A bound var inside a type or const arg (e.g.
                        // `for<'a> <F as FnOnce<(&'a mut i32,)>>::Output: 'a`)
                        // can't be reasoned about at the identity-param level,
                        // so conservatively treat the clause as giving no
                        // restriction at all.
                        if clause_arg.has_escaping_bound_vars() {
                            return no_restriction();
                        }
                    }
                }
            }
            if outlived_regions.is_empty() {
                // The bound var doesn't appear in the args at all, so the clause
                // requires the underlying type to outlive *every* region, which
                // is equivalent to a `'static` bound.
                return DenseBitSet::new_empty(clause_identity_args.len());
            }

            // The underlying type can capture any arg that's known to outlive one
            // of the bound var's positions (they're all instantiated to the same
            // region at any use site this clause applies to).
            let args_known_to_outlive = tcx.args_known_to_outlive_alias_params(alias_def_id);
            tracing::debug!(?outlived_regions, ?args_known_to_outlive);
            let mut capturable_args = DenseBitSet::new_empty(clause_identity_args.len());
            for &outlived_arg_idx in &outlived_regions {
                let (_, outliving_args) = args_known_to_outlive
                    .iter()
                    .find(|(arg_idx, _)| *arg_idx == outlived_arg_idx)
                    .unwrap();
                capturable_args.union(outliving_args);
            }
            capturable_args
        }
        // A free region (e.g. `for<a> T::Assoc<'a, 'x>: 'x`, where `'x` is free).
        // This is effectively the same as `for<'a, 'b> T::Assoc<'a, 'b>: 'b`,
        // but that only is sound if we either know that the second substituted
        // lifetime equals `'x` or if we *constrain* that lifetime to be `'x`.
        //
        // In either case, something like this doesn't work today:
        // ```ignore (illustrative)
        //  fn bar<'a, 'b>(a: &'a mut (), b: &'b ()) -> <Foo as MyTrait>::Assoc<'a, 'b> { b }
        //  fn foo<'x>()
        //  where
        //      for<'h> <Foo as MyTrait>::Assoc<'h, 'x>: 'x,
        //  {
        //      let a = &mut ();
        //      let b: &'x () = &();
        //      let val1 = rpit(a, b);
        //      let val2 = rpit(a, b);
        //      drop(val1);
        //      drop(val32);
        //  }
        // ```
        // So, we conservatively treat this as giving no restriction on which args can be captured.
        ty::ReEarlyParam(..) => no_restriction(),
        // Don't know that we actually hit this (maybe `ReError`), go ahead and be conservative.
        _ => no_restriction(),
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
    /// The current set of aliases being visited (to avoid cycles)
    pub stack: SmallVec<[DefId; 4]>,
    /// The full set of aliases that have been visited (to avoid repeated work)
    pub visited: FxHashSet<Ty<'tcx>>,
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
            ty::Alias(_, ty::AliasTy { kind, args, .. }) => {
                let tcx = self.tcx;
                let param_env = self.param_env;
                let def_id = match kind {
                    ty::AliasTyKind::Projection { def_id }
                    | ty::AliasTyKind::Inherent { def_id }
                    | ty::AliasTyKind::Opaque { def_id }
                    | ty::AliasTyKind::Free { def_id } => def_id,
                };

                // We can sometimes encounter cyclic alias types (see the `alias-liveness/cycles.rs` test).
                // If we've already seen this alias, then we just pessimistically mark all free regions as
                // live. We could actually be more precise here: for example, we could only mark the args
                // that are *different* as live, but this is the conservative approach.
                for &outer_def_id in self.stack.iter() {
                    if outer_def_id == def_id {
                        tcx.for_each_free_region(&args, |outer_r| self.visit_region(outer_r));
                        return;
                    }
                }
                // If we've seen this exact ty already (even if not in a cycle),
                // then we don't need to visit it again.
                if self.visited.contains(&ty) {
                    return;
                }
                self.stack.push(def_id);

                // For aliases other than opaques, we have to consider two
                // sources of information to identity potentially-live args:
                // - Bounds on alias item itself
                // - Outlives clauses on the current function that apply to the alias
                //
                // Each source of information *restricts* the set of potentially-live
                // args independently: only the args that can be live for *every*
                // source of information can be actually live, so we take the intersection.
                let mut capturable = tcx.live_args_for_alias_from_outlives_bounds(kind).clone();

                for clause in param_env.caller_bounds().non_trait_clauses().iter() {
                    let Some(outlives) = clause.as_type_outlives_clause() else {
                        continue;
                    };
                    capturable.intersect(&live_args_for_outlives_clause(tcx, def_id, ty, outlives));
                }
                tracing::debug!(?capturable);

                // Skip lifetime parameters that are not captured, since they do
                // not need to be live. We *cannot* skip bivariant args, because
                // they include *captured* bivariant args, which participate in
                // (invariant) outlives relationships; so they must be marked
                // as live.
                for idx in capturable.iter() {
                    args[idx as usize].visit_with(self);
                }

                // This causes us to recurse into the args of alias bounds like
                // `impl Captures<'a>` and `type Alias: Capture<a'>` and treat
                // them as live.
                for clause in tcx.item_bounds(def_id).instantiate(tcx, args).skip_norm_wip() {
                    match clause.kind().skip_binder() {
                        ty::ClauseKind::Trait(trait_clause) => {
                            trait_clause.trait_ref.args.as_slice()[1..].visit_with(self);
                        }
                        ty::ClauseKind::Projection(proj_clause) => {
                            proj_clause.projection_term.args.as_slice()[1..].visit_with(self);
                            proj_clause.term.visit_with(self);
                        }
                        _ => {}
                    }
                }

                self.stack.pop();
                self.visited.insert(ty);
            }

            _ => ty.super_visit_with(self),
        }
    }
}
