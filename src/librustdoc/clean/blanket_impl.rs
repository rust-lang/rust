use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty;
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits;
use tracing::{debug, instrument};

use crate::clean;
use crate::clean::{
    clean_middle_assoc_item, clean_middle_ty, clean_trait_ref_with_constraints, clean_ty_generics,
};
use crate::core::DocContext;

#[instrument(level = "debug", skip(cx))]
pub(crate) fn synthesize_blanket_impls(
    cx: &mut DocContext<'_>,
    item_def_id: DefId,
) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let item_ty = tcx.type_of(item_def_id);
    let param_env = ty::ParamEnv::empty();
    let typing_mode = ty::TypingMode::non_body_analysis();
    let cause = traits::ObligationCause::dummy();

    let mut blanket_impls = Vec::new();

    for trait_def_id in tcx.visible_traits() {
        if !cx.cache.effective_visibilities.is_reachable(tcx, trait_def_id)
            || cx.generated_synthetics.contains(&(item_ty.skip_binder(), trait_def_id))
        {
            continue;
        }

        for &impl_def_id in tcx.trait_impls_of(trait_def_id).blanket_impls() {
            let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);

            let ty::Param(_) = impl_trait_ref.skip_binder().self_ty().kind() else { continue };

            let infcx = tcx.infer_ctxt().with_next_trait_solver(true).build(typing_mode);

            let ocx = traits::ObligationCtxt::new(&infcx);

            let fresh_item_args = infcx.fresh_args_for_item(DUMMY_SP, item_def_id);
            let fresh_item_ty = item_ty.instantiate(tcx, fresh_item_args).skip_normalization();

            let fresh_impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
            let fresh_impl_trait_ref =
                impl_trait_ref.instantiate(tcx, fresh_impl_args).skip_normalization();

            // Constraint inference variables.
            let result = ocx.eq(&cause, param_env, fresh_impl_trait_ref.self_ty(), fresh_item_ty);
            debug_assert!(result.is_ok());

            ocx.register_obligations(traits::predicates_for_generics(
                |_, _| cause.clone(),
                |pred| ocx.normalize(&cause, param_env, pred),
                param_env,
                tcx.predicates_of(impl_def_id).instantiate(tcx, fresh_impl_args),
            ));

            if !ocx.try_evaluate_obligations().is_empty() {
                continue;
            }

            debug!("found applicable impl for trait ref {impl_trait_ref:?}");

            cx.generated_synthetics.insert((item_ty.skip_binder(), trait_def_id));

            blanket_impls.push(clean::Item {
                inner: Box::new(clean::ItemInner {
                    name: None,
                    item_id: clean::ItemId::Blanket { impl_id: impl_def_id, for_: item_def_id },
                    attrs: Default::default(),
                    stability: None,
                    kind: clean::ImplItem(Box::new(clean::Impl {
                        safety: hir::Safety::Safe,
                        generics: clean_ty_generics(cx, impl_def_id),
                        // FIXME(eddyb) compute both `trait_` and `for_` from
                        // the post-inference `trait_ref`, as it's more accurate.
                        trait_: Some(clean_trait_ref_with_constraints(
                            cx,
                            ty::Binder::dummy(
                                impl_trait_ref.instantiate_identity().skip_norm_wip(),
                            ),
                            ThinVec::new(),
                        )),
                        for_: clean_middle_ty(
                            ty::Binder::dummy(item_ty.instantiate_identity().skip_norm_wip()),
                            cx,
                            None,
                            None,
                        ),
                        items: tcx
                            .associated_items(impl_def_id)
                            .in_definition_order()
                            .filter(|item| !item.is_impl_trait_in_trait())
                            .map(|item| clean_middle_assoc_item(item, cx))
                            .collect(),
                        polarity: ty::ImplPolarity::Positive,
                        kind: clean::ImplKind::Blanket(Box::new(clean_middle_ty(
                            ty::Binder::dummy(
                                impl_trait_ref.instantiate_identity().skip_norm_wip().self_ty(),
                            ),
                            cx,
                            None,
                            None,
                        ))),
                        is_deprecated: tcx
                            .lookup_deprecation(impl_def_id)
                            .is_some_and(|deprecation| deprecation.is_in_effect()),
                    })),
                    cfg: None,
                    inline_stmt_id: None,
                }),
            });
        }
    }

    blanket_impls
}
