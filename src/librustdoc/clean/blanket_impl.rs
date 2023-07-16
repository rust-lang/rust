use crate::rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_hir as hir;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk, TyCtxtInferExt};
use rustc_infer::traits;
use rustc_middle::ty::ToPredicate;
use rustc_span::DUMMY_SP;

use super::*;

pub(crate) struct BlanketImplFinder<'a, 'tcx> {
    pub(crate) cx: &'a mut core::DocContext<'tcx>,
}

impl<'a, 'tcx> BlanketImplFinder<'a, 'tcx> {
    pub(crate) fn get_blanket_impls(&mut self, item_def_id: DefId) -> Vec<Item> {
        let cx = &mut self.cx;
        let param_env = cx.tcx.param_env(item_def_id);
        let ty = cx.tcx.type_of(item_def_id);

        trace!("get_blanket_impls({:?})", ty);
        let mut impls = Vec::new();
        for trait_def_id in cx.tcx.all_traits() {
            if !cx.cache.effective_visibilities.is_reachable(cx.tcx, trait_def_id)
                || cx.generated_synthetics.get(&(ty.skip_binder(), trait_def_id)).is_some()
            {
                continue;
            }
            // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
            let trait_impls = cx.tcx.trait_impls_of(trait_def_id);
            'blanket_impls: for &impl_def_id in trait_impls.blanket_impls() {
                trace!(
                    "get_blanket_impls: Considering impl for trait '{:?}' {:?}",
                    trait_def_id,
                    impl_def_id
                );
                let trait_ref = cx.tcx.impl_trait_ref(impl_def_id).unwrap();
                if !matches!(trait_ref.skip_binder().self_ty().kind(), ty::Param(_)) {
                    continue;
                }
                let infcx = cx.tcx.infer_ctxt().build();
                let args = infcx.fresh_args_for_item(DUMMY_SP, item_def_id);
                let impl_ty = ty.instantiate(infcx.tcx, args);
                let param_env = EarlyBinder::bind(param_env).instantiate(infcx.tcx, args);

                let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
                let impl_trait_ref = trait_ref.instantiate(infcx.tcx, impl_args);

                // Require the type the impl is implemented on to match
                // our type, and ignore the impl if there was a mismatch.
                let Ok(eq_result) = infcx.at(&traits::ObligationCause::dummy(), param_env).eq(
                    DefineOpaqueTypes::No,
                    impl_trait_ref.self_ty(),
                    impl_ty,
                ) else {
                    continue;
                };
                let InferOk { value: (), obligations } = eq_result;
                // FIXME(eddyb) ignoring `obligations` might cause false positives.
                drop(obligations);

                trace!(
                    "invoking predicate_may_hold: param_env={:?}, impl_trait_ref={:?}, impl_ty={:?}",
                    param_env,
                    impl_trait_ref,
                    impl_ty
                );
                let predicates = cx
                    .tcx
                    .predicates_of(impl_def_id)
                    .instantiate(cx.tcx, impl_args)
                    .predicates
                    .into_iter()
                    .chain(Some(ty::Binder::dummy(impl_trait_ref).to_predicate(infcx.tcx)));
                for predicate in predicates {
                    debug!("testing predicate {:?}", predicate);
                    let obligation = traits::Obligation::new(
                        infcx.tcx,
                        traits::ObligationCause::dummy(),
                        param_env,
                        predicate,
                    );
                    match infcx.evaluate_obligation(&obligation) {
                        Ok(eval_result) if eval_result.may_apply() => {}
                        Err(traits::OverflowError::Canonical) => {}
                        Err(traits::OverflowError::ErrorReporting) => {}
                        _ => continue 'blanket_impls,
                    }
                }
                debug!(
                    "get_blanket_impls: found applicable impl for trait_ref={:?}, ty={:?}",
                    trait_ref, ty
                );

                cx.generated_synthetics.insert((ty.skip_binder(), trait_def_id));

                impls.push(Item {
                    name: None,
                    attrs: Default::default(),
                    item_id: ItemId::Blanket { impl_id: impl_def_id, for_: item_def_id },
                    kind: Box::new(ImplItem(Box::new(Impl {
                        unsafety: hir::Unsafety::Normal,
                        generics: clean_ty_generics(
                            cx,
                            cx.tcx.generics_of(impl_def_id),
                            cx.tcx.explicit_predicates_of(impl_def_id),
                        ),
                        // FIXME(eddyb) compute both `trait_` and `for_` from
                        // the post-inference `trait_ref`, as it's more accurate.
                        trait_: Some(clean_trait_ref_with_bindings(
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
                        items: cx
                            .tcx
                            .associated_items(impl_def_id)
                            .in_definition_order()
                            .map(|x| clean_middle_assoc_item(x, cx))
                            .collect::<Vec<_>>(),
                        polarity: ty::ImplPolarity::Positive,
                        kind: ImplKind::Blanket(Box::new(clean_middle_ty(
                            ty::Binder::dummy(trait_ref.instantiate_identity().self_ty()),
                            cx,
                            None,
                            None,
                        ))),
                    }))),
                    cfg: None,
                    inline_stmt_id: None,
                });
            }
        }

        impls
    }
}
