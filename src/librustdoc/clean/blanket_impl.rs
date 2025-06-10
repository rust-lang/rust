use rustc_hir as hir;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk, TyCtxtInferExt};
use rustc_infer::traits;
use rustc_middle::ty::{self, TypingMode, Upcast};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use thin_vec::ThinVec;
use tracing::{debug, instrument, trace};

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

    let mut blanket_impls = Vec::new();
    for trait_def_id in tcx.all_traits() {
        if !cx.cache.effective_visibilities.is_reachable(tcx, trait_def_id)
            || cx.generated_synthetics.contains(&(ty.skip_binder(), trait_def_id))
        {
            continue;
        }
        // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
        let trait_impls = tcx.trait_impls_of(trait_def_id);
        'blanket_impls: for &impl_def_id in trait_impls.blanket_impls() {
            trace!("considering impl `{impl_def_id:?}` for trait `{trait_def_id:?}`");

            let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
            if !matches!(trait_ref.skip_binder().self_ty().kind(), ty::Param(_)) {
                continue;
            }
            let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            let args = infcx.fresh_args_for_item(DUMMY_SP, item_def_id);
            let impl_ty = ty.instantiate(tcx, args);
            let param_env = ty::ParamEnv::empty();

            let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref = trait_ref.instantiate(tcx, impl_args);

            // Require the type the impl is implemented on to match
            // our type, and ignore the impl if there was a mismatch.
            let Ok(eq_result) = infcx.at(&traits::ObligationCause::dummy(), param_env).eq(
                DefineOpaqueTypes::Yes,
                impl_trait_ref.self_ty(),
                impl_ty,
            ) else {
                continue;
            };
            let InferOk { value: (), obligations } = eq_result;
            // FIXME(eddyb) ignoring `obligations` might cause false positives.
            drop(obligations);

            let predicates = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .predicates
                .into_iter()
                .chain(Some(impl_trait_ref.upcast(tcx)));
            for predicate in predicates {
                let obligation = traits::Obligation::new(
                    tcx,
                    traits::ObligationCause::dummy(),
                    param_env,
                    predicate,
                );
                match infcx.evaluate_obligation(&obligation) {
                    Ok(eval_result) if eval_result.may_apply() => {}
                    Err(traits::OverflowError::Canonical) => {}
                    _ => continue 'blanket_impls,
                }
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
                    })),
                    cfg: None,
                    inline_stmt_id: None,
                }),
            });
        }
    }

    blanket_impls
}
