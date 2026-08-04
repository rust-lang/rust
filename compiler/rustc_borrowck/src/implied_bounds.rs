use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_infer::traits::query::MirBorrowckImpliedOutlivesBounds;
use rustc_middle::infer::canonical::{Canonical, QueryResponse};
use rustc_middle::ty::{
    self, CanonicalVarValues, GenericArg, Ty, TyCtxt, TypeVisitableExt, TypingEnv, fold_regions,
};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::solve::NoSolution;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_trait_selection::traits::query::type_op::implied_outlives_bounds::{
    compute_implied_outlives_bounds_inner, consider_implied_bounds_hack_for_ty,
};
use smallvec::SmallVec;
use tracing::instrument;

use crate::universal_regions::DefiningTy;

/// Computes the implied bounds for `body_def_id`. This is a separate query
/// as it must not reveal the hidden type of opaques defined by `body_def_id`
/// for typeck roots.
///
/// However, nested bodies are checked in the scope of their parent. This means
/// we should actually normalize opaques when computing their implied bounds.
pub(super) fn mir_borrowck_implied_outlives_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_def_id: LocalDefId,
) -> Result<
    &'tcx Canonical<'tcx, QueryResponse<'tcx, MirBorrowckImpliedOutlivesBounds<'tcx>>>,
    NoSolution,
> {
    // If we're in a typeck root we don't want to reveal any opaque types. We need to
    // make sure the caller actually checks that all our implied bounds actually hold.
    // This is not the case with the hidden types of opaque types if we're a defining-scope
    // and the caller is not.
    //
    // However, for nested bodies, we always check that they are well-formed in their
    // parent body, so for these we do want to define opaque types. Not doing so can result
    // in incorrect errors when normalizing implied bounds.
    let typing_env = if tcx.is_typeck_child(body_def_id.to_def_id()) {
        TypingEnv::post_typeck_until_borrowck(tcx, body_def_id)
    } else {
        TypingEnv::non_body_analysis(tcx, body_def_id)
    };

    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    let ocx = ObligationCtxt::new(&infcx);

    let defining_ty = DefiningTy::new(tcx, body_def_id);

    let inputs_and_output = defining_ty.inputs_and_output(tcx);
    let inputs_and_output =
        tcx.liberate_late_bound_regions(body_def_id.to_def_id(), inputs_and_output);
    let inputs_and_output = replace_erased_regions_with_placeholders(tcx, inputs_and_output);

    let mut outlives_bounds = vec![];
    // Need to return the normalized signature used to compute implied bounds back to borrowck
    // to deal with unconstrained regions due to #136547.
    let mut normalized_inputs_and_output = Vec::with_capacity(inputs_and_output.len());
    for &ty in &inputs_and_output {
        let num_registered_region_obligations = infcx.num_registered_region_obligations();
        let normalized_ty = ocx
            .deeply_normalize(&ObligationCause::dummy(), param_env, ty::Unnormalized::new_wip(ty))
            .map_err(|_| NoSolution)?;

        outlives_bounds.extend(compute_implied_outlives_bounds_inner(
            &ocx,
            param_env,
            ty,
            normalized_ty,
            DUMMY_SP,
        )?);

        outlives_bounds.extend(consider_implied_bounds_hack_for_ty(&ocx, normalized_ty, || {
            infcx.registered_region_obligations_since(num_registered_region_obligations)
        }));

        normalized_inputs_and_output.push(normalized_ty);
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
    //   the types of the locals corresponding to the inputs and outputs of the item. #136547
    if matches!(tcx.def_kind(body_def_id), DefKind::AssocFn | DefKind::AssocConst { .. }) {
        for &(ty, _) in tcx.assumed_wf_types(tcx.local_parent(body_def_id)) {
            let normalized_ty = ocx
                .deeply_normalize(
                    &ObligationCause::dummy(),
                    param_env,
                    ty::Unnormalized::new_wip(ty),
                )
                .map_err(|_| NoSolution)?;

            // We don't consider the constraints from normalizing the impl header
            // for the bevy implied bounds hack.
            let num_registered_region_obligations = infcx.num_registered_region_obligations();
            outlives_bounds.extend(compute_implied_outlives_bounds_inner(
                &ocx,
                param_env,
                normalized_ty,
                normalized_ty,
                DUMMY_SP,
            )?);

            outlives_bounds.extend(consider_implied_bounds_hack_for_ty(
                &ocx,
                normalized_ty,
                || infcx.registered_region_obligations_since(num_registered_region_obligations),
            ));
        }
    }

    let var_values = implied_bounds_query_var_values(tcx, &inputs_and_output, |r| match r.kind() {
        ty::RePlaceholder(_) => true,
        ty::ReEarlyParam(_)
        | ty::ReLateParam(_)
        | ty::ReBound(..)
        | ty::ReStatic
        | ty::ReError(_) => false,
        ty::ReVar(..) | ty::ReErased => unreachable!(),
    });
    let input_values = CanonicalVarValues { var_values: tcx.mk_args(&var_values) };

    ocx.make_canonicalized_query_response(
        input_values,
        MirBorrowckImpliedOutlivesBounds { outlives_bounds, normalized_inputs_and_output },
    )
}

/// This computes the `var_values` used by the `mir_borrowck_implied_outlives_bounds` query.
/// The old solver canonicalization does not replace early and late bound parameters,
/// so the only `var_values` we need are external regions as we don't have a shared unified
/// representation between this query and MIR borrowck.
///
/// We never late bound regions from a parent while computing implied bounds for the current item.
/// Any free region in the signature of nested body gets replaced with `'erased` at the end of HIR typeck,
/// so even if a late bound region of a parent is mentioned in our signature, it will have been erased
/// and will get represented as an external region instead.
#[instrument(level = "debug", skip(tcx, is_external_region), ret)]
pub(crate) fn implied_bounds_query_var_values<'tcx>(
    tcx: TyCtxt<'tcx>,
    unnormalized_inputs_and_output: &[Ty<'tcx>],
    mut is_external_region: impl FnMut(ty::Region<'tcx>) -> bool,
) -> SmallVec<[GenericArg<'tcx>; 8]> {
    let mut values: SmallVec<[GenericArg<'tcx>; 8]> = Default::default();

    for ty in unnormalized_inputs_and_output {
        tcx.for_each_free_region(ty, |region| {
            if is_external_region(region) {
                values.push(region.into());
            }
        });
    }

    values
}

/// This replaces all external regions in the signature of the current item with
/// a unique placeholder to collect its implied bounds. This mirrors the way MIR
/// borrowck replaces all of them with unique NLL vars.
fn replace_erased_regions_with_placeholders<'tcx>(
    tcx: TyCtxt<'tcx>,
    inputs_and_output: &[Ty<'tcx>],
) -> Vec<Ty<'tcx>> {
    debug_assert!(!inputs_and_output.has_placeholders());
    let mut next_placeholder = 0;
    inputs_and_output
        .iter()
        .map(|&ty| {
            fold_regions(tcx, ty, |r, _| match r.kind() {
                ty::ReErased => {
                    let var = ty::BoundVar::from_usize(next_placeholder);
                    next_placeholder += 1;
                    ty::Region::new_placeholder(
                        tcx,
                        ty::PlaceholderRegion::new(
                            ty::UniverseIndex::ROOT,
                            ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon },
                        ),
                    )
                }
                ty::ReEarlyParam(_)
                | ty::ReLateParam(_)
                | ty::ReBound(..)
                | ty::ReStatic
                | ty::ReError(_) => r,
                ty::ReVar(..) | ty::RePlaceholder(..) => {
                    panic!("unexpected region: {r:?}")
                }
            })
        })
        .collect()
}
