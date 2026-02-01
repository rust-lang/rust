use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty::{self, TypingMode};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits::{self, query::evaluate_obligation::InferCtxtExt};
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
    let ty = tcx.type_of(item_def_id);

    let infcx =
        tcx.infer_ctxt().with_next_trait_solver(true).build(TypingMode::non_body_analysis());

    let mut blanket_impls = Vec::new();
    for trait_def_id in tcx.visible_traits() {
        if !cx.cache.effective_visibilities.is_reachable(tcx, trait_def_id)
            || cx.generated_synthetics.contains(&(ty.skip_binder(), trait_def_id))
        {
            continue;
        }
        // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
        let trait_impls = tcx.trait_impls_of(trait_def_id);
        for &impl_def_id in trait_impls.blanket_impls() {
            let trait_ref = tcx.impl_trait_ref(impl_def_id);

            let ty::Param(_) = trait_ref.skip_binder().self_ty().kind() else { continue };

            let ocx = traits::ObligationCtxt::new(&infcx);

            let args = infcx.fresh_args_for_item(DUMMY_SP, item_def_id);
            let impl_ty = ty.instantiate(tcx, args);
            let param_env = ty::ParamEnv::empty();
            let cause = ObligationCause::dummy();

            let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = trait_ref.instantiate(tcx, impl_args);

            // Require the type the impl is implemented on to match
            // our type, and ignore the impl if there was a mismatch.
            if ocx.eq(&cause, param_env, impl_trait_ref.self_ty(), impl_ty).is_err() {
                continue;
            }

            ocx.register_obligations(traits::predicates_for_generics(
                |_, _| cause.clone(),
                param_env,
                tcx.predicates_of(impl_def_id).instantiate(tcx, impl_args),
            ));

            if !ocx.select_where_possible().is_empty() {
                continue;
            }

            debug!("found applicable impl for trait ref {trait_ref:?}");

            cx.generated_synthetics.insert((ty.skip_binder(), trait_def_id));

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
                            ty::Binder::dummy(trait_ref.instantiate_identity()),
                            ThinVec::new(),
                        )),
                        for_: clean_middle_ty(
                            ty::Binder::dummy(ty.instantiate_identity()),
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
                            ty::Binder::dummy(trait_ref.instantiate_identity().self_ty()),
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
