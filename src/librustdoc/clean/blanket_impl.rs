use crate::rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_hir as hir;
use rustc_infer::infer::{InferOk, TyCtxtInferExt};
use rustc_infer::traits;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{ToPredicate, WithConstness};
use rustc_span::DUMMY_SP;

use super::*;

crate struct BlanketImplFinder<'a, 'tcx> {
    crate cx: &'a mut core::DocContext<'tcx>,
}

impl<'a, 'tcx> BlanketImplFinder<'a, 'tcx> {
    crate fn get_blanket_impls(&mut self, item_def_id: DefId) -> Vec<Item> {
        let param_env = self.cx.tcx.param_env(item_def_id);
        let ty = self.cx.tcx.type_of(item_def_id);

        trace!("get_blanket_impls({:?})", ty);
        let mut impls = Vec::new();
        for &trait_def_id in self.cx.tcx.all_traits(()).iter() {
            if !self.cx.cache.access_levels.is_public(trait_def_id)
                || self.cx.generated_synthetics.get(&(ty, trait_def_id)).is_some()
            {
                continue;
            }
            // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
            let trait_impls = self.cx.tcx.trait_impls_of(trait_def_id);
            for &impl_def_id in trait_impls.blanket_impls() {
                trace!(
                    "get_blanket_impls: Considering impl for trait '{:?}' {:?}",
                    trait_def_id,
                    impl_def_id
                );
                let trait_ref = self.cx.tcx.impl_trait_ref(impl_def_id).unwrap();
                let is_param = matches!(trait_ref.self_ty().kind(), ty::Param(_));
                let may_apply = is_param && self.cx.tcx.infer_ctxt().enter(|infcx| {
                    let substs = infcx.fresh_substs_for_item(DUMMY_SP, item_def_id);
                    let ty = ty.subst(infcx.tcx, substs);
                    let param_env = param_env.subst(infcx.tcx, substs);

                    let impl_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
                    let trait_ref = trait_ref.subst(infcx.tcx, impl_substs);

                    // Require the type the impl is implemented on to match
                    // our type, and ignore the impl if there was a mismatch.
                    let cause = traits::ObligationCause::dummy();
                    let eq_result = infcx.at(&cause, param_env).eq(trait_ref.self_ty(), ty);
                    if let Ok(InferOk { value: (), obligations }) = eq_result {
                        // FIXME(eddyb) ignoring `obligations` might cause false positives.
                        drop(obligations);

                        trace!(
                            "invoking predicate_may_hold: param_env={:?}, trait_ref={:?}, ty={:?}",
                            param_env,
                            trait_ref,
                            ty
                        );
                        let predicates = self
                            .cx
                            .tcx
                            .predicates_of(impl_def_id)
                            .instantiate(self.cx.tcx, impl_substs)
                            .predicates
                            .into_iter()
                            .chain(Some(
                                ty::Binder::dummy(trait_ref)
                                    .without_const()
                                    .to_predicate(infcx.tcx),
                            ));
                        for predicate in predicates {
                            debug!("testing predicate {:?}", predicate);
                            let obligation = traits::Obligation::new(
                                traits::ObligationCause::dummy(),
                                param_env,
                                predicate,
                            );
                            match infcx.evaluate_obligation(&obligation) {
                                Ok(eval_result) if eval_result.may_apply() => {}
                                Err(traits::OverflowError::Canonical) => {}
                                Err(traits::OverflowError::ErrorReporting) => {}
                                _ => {
                                    return false;
                                }
                            }
                        }
                        true
                    } else {
                        false
                    }
                });
                debug!(
                    "get_blanket_impls: found applicable impl: {} for trait_ref={:?}, ty={:?}",
                    may_apply, trait_ref, ty
                );
                if !may_apply {
                    continue;
                }

                self.cx.generated_synthetics.insert((ty, trait_def_id));

                impls.push(Item {
                    name: None,
                    attrs: Default::default(),
                    visibility: Inherited,
                    def_id: ItemId::Blanket { impl_id: impl_def_id, for_: item_def_id },
                    kind: box ImplItem(Impl {
                        span: Span::new(self.cx.tcx.def_span(impl_def_id)),
                        unsafety: hir::Unsafety::Normal,
                        generics: (
                            self.cx.tcx.generics_of(impl_def_id),
                            self.cx.tcx.explicit_predicates_of(impl_def_id),
                        )
                            .clean(self.cx),
                        // FIXME(eddyb) compute both `trait_` and `for_` from
                        // the post-inference `trait_ref`, as it's more accurate.
                        trait_: Some(trait_ref.clean(self.cx)),
                        for_: ty.clean(self.cx),
                        items: self
                            .cx
                            .tcx
                            .associated_items(impl_def_id)
                            .in_definition_order()
                            .collect::<Vec<_>>()
                            .clean(self.cx),
                        negative_polarity: false,
                        synthetic: false,
                        blanket_impl: Some(box trait_ref.self_ty().clean(self.cx)),
                    }),
                    cfg: None,
                });
            }
        }
        impls
    }
}
