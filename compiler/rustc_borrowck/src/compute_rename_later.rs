// TODO: add description later

use rustc_infer::infer::canonical::Canonical;
use rustc_infer::infer::{TyCtxtInferExt, canonical};
use rustc_infer::traits::ObligationCause;
use rustc_infer::traits::query::OutlivesBound;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_trait_selection::traits::query::type_op::implied_outlives_bounds::compute_implied_outlives_bounds_inner;

use crate::hir::def::DefKind;
use crate::ty::solve::NoSolution;
use crate::ty::{CanonicalVarValues, GenericArg, GenericArgKind, GenericArgs, Region, RegionKind};
use crate::universal_regions::{
    compute_inputs_and_output_non_nll, defining_ty_non_nll,
    for_each_late_bound_region_in_recursive_scope,
};
use crate::{
    LocalDefId, ParamEnv, RegionVariableOrigin, SmallVec, Ty, TypingMode, debug, fold_regions, ty,
};

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { compute_outlives_bounds_rename, ..*p };
}

fn compute_outlives_bounds_rename<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_def: LocalDefId,
) -> Result<
    &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
    NoSolution,
> {
    debug!("enter compute_outlives_bounds_rename");
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new(&infcx);
    let param_env = ParamEnv::empty();
    let defining_ty = defining_ty_non_nll(&infcx, mir_def);
    let defining_ty_def_id = defining_ty.def_id().expect_local();
    let mut late_bound_region: Vec<Region<'_>> = vec![];

    let unnormalized_input_output_tys =
        compute_inputs_and_output_non_nll(&infcx, mir_def, defining_ty);

    let unnormalized_input_output_tys = tcx
        .liberate_late_bound_regions(defining_ty_def_id.to_def_id(), unnormalized_input_output_tys);

    let span = tcx.def_span(defining_ty_def_id);
    let mut outlives_bounds: Vec<OutlivesBound<'tcx>> = vec![];
    let mut norm_sig_tys: Vec<Ty<'_>> = vec![];

    for ty in unnormalized_input_output_tys {
        debug!("the ty for input and output are {:?}", ty);
        debug!("the kind of type is {:?}", ty.kind());
        // Replace erased regions with fresh region variables.
        let ty = fold_regions(tcx, ty, |re, _dbi| match re.kind() {
            ty::ReErased => infcx.next_region_var(RegionVariableOrigin::Misc(span)),
            _ => re,
        });

        // We add implied bounds from both the unnormalized and normalized ty.
        // See issue #87748
        if let Ok(bounds) = compute_implied_outlives_bounds_inner(&ocx, param_env, ty, span, false)
        {
            outlives_bounds.extend(bounds);
        }

        let Ok(norm_ty) =
            ocx.deeply_normalize(&ObligationCause::dummy_with_span(span), param_env, ty)
        else {
            // Even if deeply normalize returns no solution, we still need to store the ty for canonicalization later.
            norm_sig_tys.push(ty);
            continue;
        };

        // Currently `implied_outlives_bounds` will normalize the provided
        // `Ty`, despite this it's still important to normalize the ty ourselves
        // as normalization may introduce new region variables (#136547).
        //
        // If we do not add implied bounds for the type involving these new
        // region variables then we'll wind up with the normalized form of
        // the signature having not-wf types due to unsatisfied region
        // constraints.
        //
        // Note: we need this in examples like
        // ```
        // trait Foo {
        //   type Bar;
        //   fn foo(&self) -> &Self::Bar;
        // }
        // impl Foo for () {
        //   type Bar = ();
        //   fn foo(&self) -> &() {}
        // }
        // ```
        // Both &Self::Bar and &() are WF
        if ty != norm_ty {
            if let Ok(bounds) =
                compute_implied_outlives_bounds_inner(&ocx, param_env, norm_ty, span, false)
            {
                outlives_bounds.extend(bounds);
            }
        }

        // Collect late bound region
        if let ty::Ref(region, _, _) = ty.kind() {
            if let RegionKind::ReLateParam(..) = region.kind() {
                late_bound_region.push(*region);
            }
        }

        norm_sig_tys.push(norm_ty);
    }

    // Add implied bounds from impl header.
    //
    // We don't use `assumed_wf_types` to source the entire set of implied bounds for
    // a few reasons:
    // - `DefiningTy` for closure has the `&'env Self` type while `assumed_wf_types` doesn't
    // - We compute implied bounds from the unnormalized types in the `DefiningTy` but do not
    //   do so for types in impl headers
    // - We must compute the normalized signature and then compute implied bounds from that
    //   in order to connect any unconstrained region vars created during normalization to
    //   the types of the locals corresponding to the inputs and outputs of the item. (#136547)
    if matches!(tcx.def_kind(defining_ty_def_id), DefKind::AssocFn | DefKind::AssocConst) {
        for &(ty, _) in tcx.assumed_wf_types(tcx.local_parent(defining_ty_def_id)) {
            // Replace erased regions with fresh region variables.
            let ty = fold_regions(tcx, ty, |re, _dbi| match re.kind() {
                ty::ReErased => infcx.next_region_var(RegionVariableOrigin::Misc(span)),
                _ => re,
            });

            let Ok(norm_ty) =
                ocx.deeply_normalize(&ObligationCause::dummy_with_span(span), param_env, ty)
            else {
                continue;
            };

            // We currently add implied bounds from the normalized ty only.
            // This is more conservative and matches wfcheck behavior.
            if let Ok(bounds) =
                compute_implied_outlives_bounds_inner(&ocx, param_env, norm_ty, span, false)
            {
                outlives_bounds.extend(bounds);
            }
        }
    }
    // TODO: clean up all these messy bunch together

    // Get early bound params.
    let typeck_root_def_id = tcx.typeck_root_def_id(mir_def.to_def_id());

    let early_bound_params = GenericArgs::identity_for_item(tcx, typeck_root_def_id);
    let mut early_bound_region: SmallVec<[GenericArg<'_>; 8]> = Default::default();
    let mut early_bound_non_region: SmallVec<[GenericArg<'_>; 8]> = Default::default();

    for param in early_bound_params {
        match param.kind() {
            GenericArgKind::Lifetime(_) => early_bound_region.push(param),
            _ => early_bound_non_region.push(param),
        }
    }

    let mut var_value: SmallVec<[GenericArg<'_>; 8]> = early_bound_non_region;

    for param in early_bound_region {
        var_value.push(param);
    }

    // Collect late bound region for closure, coroutine, or inline-const.
    // TODO: remove this?
    if mir_def.to_def_id() != typeck_root_def_id {
        for_each_late_bound_region_in_recursive_scope(tcx, tcx.local_parent(mir_def), |r| {
            var_value.push(GenericArg::from(r));
        });
    }

    // var value = [generic param, early bound region, late bound region, sig tys]
    // Make generic param goes before region to match with the call site in free_region_relations.

    for region in &late_bound_region {
        var_value.push(GenericArg::from(*region))
    }

    for ty in &norm_sig_tys {
        var_value.push(GenericArg::from(*ty))
    }

    // TODO: see if either the early and late bound region are universal. If it is,
    // then we need a separate var value for it? test closures too.

    let var_values: CanonicalVarValues<TyCtxt<'_>> =
        CanonicalVarValues { var_values: tcx.mk_args(var_value.as_slice()) };

    debug!("var value in query is {:?}", var_values);

    ocx.make_canonicalized_query_response(var_values, outlives_bounds)
}
